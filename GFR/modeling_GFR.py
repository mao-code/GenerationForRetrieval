# coding=utf-8
# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch GFR model."""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_mamba_ssm_available,
)
from GFR.configuration_GFR import GFRConfig


if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GFRConfig"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->GFR
class GFRRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GFRRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(GFRRMSNorm)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Copied from transformers.models.GFR.modeling_GFR with Zamba->GFR
class GFRHybridDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.has_previous_state = False  # only used by mamba
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.n_mamba_heads = config.n_mamba_heads
        self.conv_states = []
        self.ssm_states = []
        self.transformer_layers = [] # Will hold indices of transformer (attention) layers.
        self._modules = {}
        self._parameters = {}
        self._buffers = {}

        # Loop over all hidden layers.
        for i in range(config.num_hidden_layers):
            self.conv_states.append(
                torch.zeros(batch_size, self.intermediate_size, self.conv_kernel_size, device=device, dtype=dtype)
            )
            cache_shape = (
                batch_size,
                self.n_mamba_heads,
                self.intermediate_size // self.n_mamba_heads,
                self.ssm_state_size,
            )
            self.ssm_states.append(torch.zeros(cache_shape, device=device, dtype=dtype))
            # Mark transformer layers based on the block structure.
            # In our model, transformer layers are at positions 0 and 4 in each block.
            if i % config.num_layers_per_block in [0, 4]:
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    # Copied from transformers.models.jamba.modeling_jamba.HybridMambaAttentionDynamicCache.update
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # Copied from transformers.models.jamba.modeling_jamba.HybridMambaAttentionDynamicCache.reorder_cache
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    # Copied from transformers.models.jamba.modeling_jamba.HybridMambaAttentionDynamicCache.get_seq_length
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("GFRHybridDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("GFRHybridDynamicCache does not have a legacy cache equivalent.")

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # if attention_mask is not None:
    #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #     attn_weights = attn_weights + causal_mask
    if attention_mask is not None:
        # If the attention_mask is 2D, convert it to 4D by unsqueezing dimensions.
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class GFRAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. 
    Modified to use sliding window attention: Longformer and "Generating Long Sequences with Sparse Transformers". (This feature will be activated only if you provide a `attention_mask`. If not, it will default to the standard multi-head attention.)
    
    Adapted from transformers.models.mistral.modeling_mistral.MistralAttention
    """

    def __init__(self, config: GFRConfig, layer_idx: int, concat_input: Optional[bool] = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_hidden_size = config.hidden_size * (2 if concat_input else 1)
        self.head_dim = (2 if concat_input else 1) * config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.scaling = (self.head_dim / (2 if concat_input else 1)) ** -0.5
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.attention_hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.attention_hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.attention_hidden_size, config.num_key_value_heads * self.head_dim, bias=False)

        # For residual, we need to project the output with the same size as the input
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, self.attention_hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[GFRHybridDynamicCache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        # This view splits the last dimension into (num_heads, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class GFRMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)

    This version supports an optional concatenated input. When `concat_input=True`, it expects 
    the input to have dimension 2*hidden_size (as used in the first mamba block of the overall GFR model);
    otherwise, it expects an input of size hidden_size.
    
    The gated linear projection splits the output into two halves (for the state and for the gating), 
    and then each mamba head is processed independently.

    OLD:
    This module differs from `transformers.models.mamba.modeling_mamba.MambaMixer` in two ways:
    - Added multi-head: the output of `self.in_proj` is split into `self.n_mamba_heads` heads, and each head
    undergoes an independent forward pass, identical to the original `MambaMixer`, up until the pre-activations of
    `self.out_proj`. The pre-activations, coming from different mamba heads, are then concatenated and fed into `self.out_proj`.
    """

    def __init__(self, config: GFRConfig, layer_idx, concat_input: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.n_mamba_heads = config.n_mamba_heads
        self.mamba_head_dim = self.intermediate_size // self.n_mamba_heads
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        
        self.concat_input = concat_input

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.hidden_mamba_act
        self.act = ACT2FN[config.hidden_mamba_act]

        self.use_fast_kernels = config.use_mamba_kernels

        # Determine the number of input channels depending on whether we concatenate.
        in_features = config.hidden_size * (2 if self.concat_input else 1)
        # Projection of the input hidden states
        # From (batch_size, seq_length, in_features) to (batch_size, seq_length, intermediate_size * 2)
        self.in_proj = nn.Linear(in_features, self.intermediate_size * 2, bias=self.use_bias)

        # projection of the input hidden states
        # self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        # weight associated to the selective projection used to make dt, B and C input dependent
        # each mamba head is processed independently
        self.x_proj_weight = nn.Parameter(
            (
                torch.zeros(
                    self.n_mamba_heads,
                    self.time_step_rank + self.ssm_state_size * 2,
                    self.mamba_head_dim,
                )
            )
        )
        # time step projection (discretization)
        self.dt_proj_weight = nn.Parameter(
            (torch.zeros(self.n_mamba_heads, self.mamba_head_dim, self.time_step_rank) - 0.5)
            * 2
            / self.time_step_rank**0.5
        )
        self.dt_proj_bias = nn.Parameter(torch.zeros(self.n_mamba_heads, self.mamba_head_dim))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A).reshape(self.n_mamba_heads, self.mamba_head_dim, -1))
        self.D = nn.Parameter(torch.ones(self.n_mamba_heads, self.mamba_head_dim))

        # If concatenate input, we need to project the input to the correct size (the same size of the input).
        output_hidden_size = self.hidden_size * (2 if self.concat_input else 1)
        self.out_proj = nn.Linear(self.intermediate_size, output_hidden_size, bias=self.use_bias)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model config"
            )

    def cuda_kernels_forward(
        self, hidden_states: torch.Tensor, cache_params: GFRHybridDynamicCache = None, attention_mask=None
    ):
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state and seq_len == 1

        # 1. Gated linear projection (2 gates here, so 2*intermidiate_size)
        # Swaps dimensions 1 and 2
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        hidden_states, gate = projected_states.view(batch_size, -1, 2, seq_len).chunk(2, dim=2)
        hidden_states = hidden_states.squeeze(2).contiguous()
        gate = gate.squeeze(2)
        gate = gate.reshape(batch_size, self.n_mamba_heads, -1, seq_len).transpose(0, 1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if use_precomputed_states:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_states[self.layer_idx].to(hidden_states.dtype),
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if attention_mask is not None and not torch.all(attention_mask == 1):
                hidden_states = hidden_states * attention_mask.unsqueeze(1)
            if cache_params is not None:
                conv_states = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_states[self.layer_idx].copy_(conv_states)
            hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)
            if attention_mask is not None and not torch.all(attention_mask == 1):
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. SSM sequence transformation
        # 3.a. input varying initialization of time_step, B and C

        hidden_states = hidden_states.reshape(-1, self.n_mamba_heads, self.mamba_head_dim, seq_len).transpose(0, 1)
        ssm_parameters = (self.x_proj_weight[:, None, :, :] @ hidden_states).transpose(-1, -2)

        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        discrete_time_step = self.dt_proj_weight[:, None] @ time_step.transpose(-1, -2)

        A = -torch.exp(self.A_log.float())

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = self.dt_proj_bias.float() if self.dt_proj_bias is not None else None
        scan_outputs = torch.empty((batch_size, 0, seq_len), device=hidden_states.device, dtype=hidden_states.dtype)

        if use_precomputed_states:
            for n in range(self.n_mamba_heads):
                scan_outputs_ = selective_state_update(
                    cache_params.ssm_states[self.layer_idx][:, n],
                    hidden_states[n, ..., 0],
                    discrete_time_step[n, ..., 0],
                    A[n],
                    B[n, :, 0],
                    C[n, :, 0],
                    self.D[n],
                    gate[n, ..., 0],
                    time_proj_bias[n],
                    dt_softplus=True,
                ).unsqueeze(-1)
                scan_outputs = torch.cat((scan_outputs, scan_outputs_), dim=1)

        else:
            ssm_state = torch.empty(
                (batch_size, 0, self.mamba_head_dim, self.ssm_state_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            for n in range(self.n_mamba_heads):
                scan_outputs_, ssm_state_ = selective_scan_fn(
                    hidden_states[n],
                    discrete_time_step[n],
                    A[n],
                    B[n].transpose(1, 2),
                    C[n].transpose(1, 2),
                    self.D[n].float(),
                    gate[n],
                    time_proj_bias[n],
                    delta_softplus=True,
                    return_last_state=True,
                )
                scan_outputs = torch.cat((scan_outputs, scan_outputs_), dim=1).contiguous()
                ssm_state = torch.cat((ssm_state, ssm_state_.unsqueeze(1)), dim=1)
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def slow_forward(self, input_states, cache_params: GFRHybridDynamicCache = None, attention_mask=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)

        hidden_states, gate = projected_states.view(batch_size, -1, 2, seq_len).chunk(2, dim=2)
        hidden_states = hidden_states.squeeze(2).contiguous()
        gate = gate.squeeze(2)
        gate = gate.reshape(batch_size, self.n_mamba_heads, -1, seq_len).transpose(0, 1)

        use_cache = isinstance(cache_params, GFRHybridDynamicCache)
        # 2. Convolution sequence transformation
        if use_cache and cache_params.ssm_states[self.layer_idx].shape[0] == batch_size:
            if self.training:
                # In training mode, we don't want to perform in-place operations on ssm_state so we can compute the backwards pass
                ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            else:
                ssm_state = cache_params.ssm_states[self.layer_idx]

            ssm_state = ssm_state.to(hidden_states.device)

            if (
                cache_params.has_previous_state
                and seq_len == 1
                and cache_params.conv_states[self.layer_idx].shape[0] == batch_size
            ):
                conv_state = cache_params.conv_states[self.layer_idx]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
            else:
                if attention_mask is not None and not torch.all(attention_mask == 1):
                    hidden_states = hidden_states * attention_mask[:, -hidden_states.shape[-1] :].unsqueeze(1)
                conv_state = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
                if attention_mask is not None and not torch.all(attention_mask == 1):
                    hidden_states = hidden_states * attention_mask[:, -hidden_states.shape[-1] :].unsqueeze(1)
        else:
            ssm_state = torch.zeros(
                (batch_size, self.n_mamba_heads, self.mamba_head_dim, self.ssm_state_size),
                device=hidden_states.device,
                dtype=dtype,
            )
            if attention_mask is not None and not torch.all(attention_mask == 1):
                hidden_states = hidden_states * attention_mask.unsqueeze(1)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
            if attention_mask is not None and not torch.all(attention_mask == 1):
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        hidden_states = hidden_states.reshape(-1, self.n_mamba_heads, self.mamba_head_dim, seq_len).transpose(0, 1)
        ssm_parameters = (self.x_proj_weight[:, None, :, :] @ hidden_states).transpose(-1, -2)

        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = (self.dt_proj_weight[:, None] @ time_step.transpose(-1, -2)) + self.dt_proj_bias[
            :, None, :, None
        ]

        discrete_time_step = nn.functional.softplus(discrete_time_step)

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())
        discrete_A = torch.exp(A[:, None, :, None, :] * discrete_time_step[:, :, :, :, None])
        discrete_B = discrete_time_step[:, :, :, :, None] * B[:, :, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, :, None].float()
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, :, i, :].transpose(0, 1) * ssm_state + deltaB_u[:, :, :, i, :].transpose(0, 1)
            scan_output = torch.matmul(ssm_state.transpose(0, 1).to(dtype), C[:, :, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + (hidden_states * self.D[:, None, :, None])
        scan_output = scan_output * self.act(gate)

        if use_cache:
            cache_params.ssm_states[self.layer_idx] = ssm_state

        # 4. Final linear projection
        contextualized_states = self.out_proj(
            scan_output.transpose(0, 1).reshape(batch_size, -1, seq_len).transpose(1, 2)
        )
        return contextualized_states

    def forward(self, hidden_states, cache_params: GFRHybridDynamicCache = None, attention_mask=None):
        if self.use_fast_kernels:
            if not is_fast_path_available or "cuda" not in self.x_proj_weight.device.type:
                raise ValueError(
                    "Fast Mamba kernels are not available. Make sure to they are installed and that "
                    "the mamba module is on a CUDA device. lease run 'pip install causal-conv1d>=1.2.0' "
                    "and 'pip install mamba-ssm', or set use_mamba_kernels=False in the model's config."
                )
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask=attention_mask)
        return self.slow_forward(hidden_states, cache_params, attention_mask=attention_mask)


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->GFR
class GFRMLP(nn.Module):
    def __init__(self, config: GFRConfig, concat_input: Optional[bool] = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size * (2 if concat_input else 1)
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# Transformer Block
class GFRAttentionDecoderLayer(nn.Module):
    def __init__(self, config: GFRConfig, layer_idx: Optional[int] = None, concat_input: Optional[bool] = False):
        super().__init__()
        self.self_attn = GFRAttention(config, layer_idx, concat_input)
        norm_hidden_size = config.hidden_size * (2 if concat_input else 1)
        
        self.feed_forward = GFRMLP(config, concat_input=concat_input)
        self.input_layernorm = GFRRMSNorm(norm_hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = GFRRMSNorm(norm_hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx
        self.concat_input = concat_input

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[GFRHybridDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): output of previous Mamba layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output of shape `(batch, seq_len, embed_dim)`.
                This is concatenated with `hidden_states` (which is the output of the previous (mamba) layer). The
                concatenated tensor is then used as input of the pre-attention RMSNorm
                (see fig. 2 in https://arxiv.org/pdf/2405.16712).
            layer_idx (`int`): layer_idx in the forward pass. Used to distinguish GFR's tied transformer layers.
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`GFRHybridDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """        
        if self.concat_input and original_hidden_states is not None:
            hidden_states = torch.concatenate([hidden_states, original_hidden_states], dim=-1)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            layer_idx=self.layer_idx,
            attention_mask=causal_mask, # pass causal mask to attention not padding mask
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # residual connection after attention
        hidden_states = residual + hidden_states

        # feed-forward (MLP)
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

# Mamba Block
class GFRMambaDecoderLayer(nn.Module):
    def __init__(self, config: GFRConfig, layer_idx: int, concat_input: Optional[bool] = False):
        super().__init__()
        self.mamba = GFRMambaMixer(config=config, layer_idx=layer_idx, concat_input=concat_input)
        norm_hidden_size = config.hidden_size * (2 if concat_input else 1)
        self.input_layernorm = GFRRMSNorm(norm_hidden_size, eps=config.rms_norm_eps)

        self.pre_ff_layernorm = GFRRMSNorm(norm_hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = GFRMLP(config, concat_input=concat_input)

        self.layer_idx = layer_idx
        self.concat_input = concat_input

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[GFRHybridDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`GFRHybridDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """
        if self.concat_input and original_hidden_states is not None:
            hidden_states = torch.concatenate([hidden_states, original_hidden_states], dim=-1)

        # Mamba
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_value,
            attention_mask=attention_mask,
        ) # Back to hidden_size
        self_attn_weights = None

        # residual connection after mamba
        hidden_states = residual + hidden_states

        # feed-forward (MLP)
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs

GFR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GFRConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare GFR Model outputting raw hidden-states without any specific head on top.",
    GFR_START_DOCSTRING,
)
class GFRPreTrainedModel(PreTrainedModel):
    config_class = GFRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GFRAttentionDecoderLayer", "GFRMambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True  # Note: only supports GFRHybridDynamicCache
    _is_stateful = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GFRMambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            module.x_proj_weight.data.normal_(mean=0.0, std=std)
            dt_init_std = self.config.mamba_dt_rank**-0.5
            nn.init.uniform_(module.dt_proj_weight, -dt_init_std, dt_init_std)

            mamba_head_dim = self.config.mamba_expand * self.config.hidden_size // self.config.n_mamba_heads
            dt = torch.exp(
                torch.rand(self.config.n_mamba_heads, mamba_head_dim)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                module.dt_proj_bias.copy_(inv_dt)
            module.dt_proj_bias._no_reinit = True

    @classmethod
    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        hard_check_only: bool = False,
        check_device_map: bool = False,
    ):
        """
        Overloads `PreTrainedModel._check_and_enable_flash_attn_2` so as to DISABLE Flash Attention 2 by default on GFR models.
        Flash attention 2 is currently not supported in the HuggingFace implementation of GFR v1.
        """
        config = super()._check_and_enable_flash_attn_2(
            config, torch_dtype, device_map, hard_check_only=hard_check_only, check_device_map=check_device_map
        )

        # if using the default path -> swap sdpa by eager
        if not hard_check_only and config._attn_implementation == "flash_attention_2":
            config._attn_implementation = "eager"

        return config

GFR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`GFRHybridDynamicCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            A GFRHybridDynamicCache object containing pre-computed hidden-states (keys and values in the
            self-attention blocks and convolution and ssm states in the mamba blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
            Key and value cache tensors have shape `(batch_size, num_heads, seq_len, head_dim)`.
            Convolution and ssm states tensors have shape `(batch_size, d_inner, d_conv)` and
            `(batch_size, d_inner, d_state)` respectively.
            See the `GFRHybridDynamicCache` class for more details.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

class GFRModel(GFRPreTrainedModel):
    """
    Base GFR model that returns the final hidden states.
    
    This model implements the layer-by-layer transformer backbone.
    It accepts input_ids and returns the hidden states.

    A "layer-by-layer" variant of your original GFRModel. 

    Instead of creating full blocks via GFRBlock, we explicitly create each sub-layer in sequence:
      - Transformer 1 (T1)
      - Mamba 1 (M1)
      - Mamba 2 (M2)
      - Mamba 3 (M3)
         - After M3, add skip connection from T1 output
      - Transformer 2 (T2, concat_input=True)
      - Mamba 4 (M4)
      - Mamba 5 (M5)
      - Mamba 6 (M6)

    Each "block" of 8 sub-layers is repeated `config.num_hidden_blocks` times, if you so desire.
    """
    def __init__(self, config: "GFRConfig"):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings: token embeddings and token-type embeddings.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        # Add tied weights keys: indicate that lm_head.weight should share the same weights as embed_tokens.weight.
        self._tied_weights_keys = ["embed_tokens.weight"]

        # Build the backbone with repeated blocks. Each block consists of 8 sub-layers.
        self.layers = nn.ModuleList()
        self.num_hidden_blocks = config.num_hidden_blocks
        self.num_layers_per_block = config.num_layers_per_block 
        self.num_hidden_layers = config.num_hidden_layers  # total number of sub-layers
        for block_idx in range(self.num_hidden_blocks):
            base_idx = block_idx * self.num_layers_per_block
            # 1) Transformer block T1 (no concatenation)
            self.layers.append(
                GFRAttentionDecoderLayer(
                    config,
                    layer_idx=base_idx + 0,
                    concat_input=False
                )
            )
            # 2) Mamba 1
            self.layers.append(
                GFRMambaDecoderLayer(
                    config,
                    layer_idx=base_idx + 1,
                    concat_input=False
                )
            )
            # 3) Mamba 2
            self.layers.append(
                GFRMambaDecoderLayer(
                    config,
                    layer_idx=base_idx + 2,
                    concat_input=False
                )
            )
            # 4) Mamba 3
            self.layers.append(
                GFRMambaDecoderLayer(
                    config,
                    layer_idx=base_idx + 3,
                    concat_input=False
                )
            )
            # 5) Transformer block T2 (with concatenation)
            self.layers.append(
                GFRAttentionDecoderLayer(
                    config,
                    layer_idx=base_idx + 4,
                    concat_input=True
                )
            )
            #  Linear projection after T2 to match hidden size
            self.layers.append(
                nn.Linear(
                    2 * config.hidden_size,
                    config.hidden_size,
                    bias=True
                )
            )
            # 6) Mamba 4
            self.layers.append(
                GFRMambaDecoderLayer(
                    config,
                    layer_idx=base_idx + 5,
                    concat_input=False
                )
            )
            # 7) Mamba 5
            self.layers.append(
                GFRMambaDecoderLayer(
                    config,
                    layer_idx=base_idx + 6,
                    concat_input=False
                )
            )
            # 8) Mamba 6
            self.layers.append(
                GFRMambaDecoderLayer(
                    config,
                    layer_idx=base_idx + 7,
                    concat_input=False
                )
            )

        # Final normalization layer.
        self.final_layernorm = GFRRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # A flag for gradient checkpointing.
        self.gradient_checkpointing = False
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward("GFR Model forward", "GFR_INPUTS_DOCSTRING")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["GFRHybridDynamicCache"] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Processes the input and returns the final hidden states.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # If gradient checkpointing is enabled, we cannot use cache.
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # Exactly one of {input_ids, inputs_embeds} must be provided.
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify either input_ids or inputs_embeds (but not both).")

        # Embed the inputs.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)

        # If token_type_ids is missing, default to zeros.
        if token_type_ids is None and input_ids is not None:
            token_type_ids = torch.zeros_like(input_ids)
        # token_type_embeds = self.token_type_embeddings(token_type_ids)  # (batch, seq_len, hidden_size)

        # Sum token and token-type embeddings.
        # hidden_states = inputs_embeds + token_type_embeds
        hidden_states = inputs_embeds
        original_hidden_states = hidden_states.clone()

        batch_size, seq_len, _ = hidden_states.shape
        if cache_position is None:
            cache_position = torch.arange(seq_len, device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if use_cache and past_key_values is None:
            logger.warning_once(
                "GFR requires an initialized `GFRHybridDynamicCache` to return a cache. None was provided."
            )

        # Build (or update) the causal mask as needed.
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        layer_index = 0
        # Iterate over each block.
        for block_idx in range(self.num_hidden_blocks):
            # ----- 1) Transformer T1 -----
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    self.layers[layer_index].__call__,
                    hidden_states,
                    original_hidden_states,
                    attention_mask,
                    causal_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position
                )
            else:
                layer_outputs = self.layers[layer_index](
                    hidden_states,
                    original_hidden_states=original_hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
            T1_out = layer_outputs[0]
            if output_attentions and layer_outputs[1] is not None:
                all_attentions += (layer_outputs[1],)
            layer_index += 1
            hidden_states = T1_out

            # ----- 2) Mamba 1 -----
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_index](
                hidden_states,
                original_hidden_states=original_hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and layer_outputs[1] is not None:
                all_attentions += (layer_outputs[1],)
            layer_index += 1

            # ----- 3) Mamba 2 -----
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_index](
                hidden_states,
                original_hidden_states=original_hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and layer_outputs[1] is not None:
                all_attentions += (layer_outputs[1],)
            layer_index += 1

            # ----- 4) Mamba 3 -----
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_index](
                hidden_states,
                original_hidden_states=original_hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and layer_outputs[1] is not None:
                all_attentions += (layer_outputs[1],)
            layer_index += 1

            # Save Mamba 3 output for later skip connection.
            mamba3_output_hidden_states = hidden_states

            # ----- 5) Transformer T2 (concat_input=True) -----
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_index](
                hidden_states,
                original_hidden_states=original_hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and layer_outputs[1] is not None:
                all_attentions += (layer_outputs[1],)
            layer_index += 1

            # Linear projection after T2.
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_index](hidden_states)
            hidden_states = layer_outputs
            layer_index += 1

            # Add skip connection from Mamba 3's output.
            hidden_states = hidden_states + mamba3_output_hidden_states

            # ----- 6) Mamba 4 -----
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_index](
                hidden_states,
                original_hidden_states=original_hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and layer_outputs[1] is not None:
                all_attentions += (layer_outputs[1],)
            layer_index += 1

            # ----- 7) Mamba 5 -----
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_index](
                hidden_states,
                original_hidden_states=original_hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and layer_outputs[1] is not None:
                all_attentions += (layer_outputs[1],)
            layer_index += 1

            # ----- 8) Mamba 6 -----
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_index](
                hidden_states,
                original_hidden_states=original_hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and layer_outputs[1] is not None:
                all_attentions += (layer_outputs[1],)
            layer_index += 1

        # Final normalization.
        hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values is not None and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        if not return_dict:
            return (hidden_states, past_key_values) if use_cache else (hidden_states,)
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )
        
    # Copied from transformers.models.jamba.modeling_jamba.JambaModel._update_causal_mask
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class GFRForCausalLM(GFRPreTrainedModel, GenerationMixin):
    """
    GFR model with a causal language modeling head.
    
    This model wraps the GFRModel backbone and adds an LM head that projects
    the final hidden states at every position to vocabulary logits.
    """
    def __init__(self, config: "GFRConfig"):
        super().__init__(config)
        self.config = config
        self.model = GFRModel(config)
        self.vocab_size = config.vocab_size
        # LM head: projects hidden states to logits over the vocabulary.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._tied_weights_keys = ["lm_head.weight", *self.model._tied_weights_keys]
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")   
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["GFRHybridDynamicCache"] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for causal language modeling.
        
        The backbone outputs hidden states which are passed through the LM head
        to produce logits over the vocabulary.
        """
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]) # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwitten -- has a unique cache type, `GFRHybridDynamicCache`

        empty_past_kv = past_key_values is None

        # Omit tokens covered by past_key_values
        if not empty_past_kv:
            # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
            # Exception 1: when passing input_embeds, input_ids may be missing entries
            # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
            # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
            #              (we can't check exception 3 while compiling)
            if (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = GFRHybridDynamicCache(
                self.config, input_ids.shape[0], dtype=self.dtype, device=self.device
            )

        # If past_key_values is not a GFRHybridDynamicCache, set it forcely
        if not isinstance(past_key_values, GFRHybridDynamicCache):
            print("Got a non-GFR cache, set it forcely")
            past_key_values = GFRHybridDynamicCache(
                self.config, input_ids.shape[0], dtype=self.dtype, device=self.device
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "logits_to_keep": self.config.num_logits_to_keep,
                "cache_position": cache_position,
            }
        )
        return model_inputs

class GFRModelWithTokenTypes(GFRModel):
    def __init__(self, config: "GFRConfig"):
        super().__init__(config)
        # Add token type embeddings with a small vocabulary (e.g., 2 types: 0 for document, 1 for query)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        # Initialize the token type embeddings to zeros so that they don't disturb the pre-trained weights initially.
        nn.init.zeros_(self.token_type_embeddings.weight)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["GFRHybridDynamicCache"] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, "BaseModelOutputWithPast"]:
        # If inputs_embeds is not provided, compute them from input_ids.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)

            if token_type_ids is not None:
                token_type_embeds = self.token_type_embeddings(token_type_ids)
                inputs_embeds = inputs_embeds + token_type_embeds

        # Call the parent forward method with the modified embeddings.
        # We pass token_type_ids as None because we already added them.
        return super().forward(
            input_ids=None,
            token_type_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

class GFRForSequenceScoring(GFRPreTrainedModel):
    """
    GFR model for sequence scoring.
    
    This model wraps the GFRModel backbone and adds a score head that uses
    the [CLS] token (first token) hidden state for scoring.
    """
    def __init__(self, config: "GFRConfig"):
        super().__init__(config)
        self.config = config
        self.gfr = GFRModelWithTokenTypes(config)

        self.token_type_embedding = nn.Embedding(2, config.hidden_size)
        # Score head: projects the [CLS] token's hidden state to class logits.
        self.score_head = nn.Linear(config.hidden_size, 1, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.gfr.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        return self.gfr.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["GFRHybridDynamicCache"] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass for sequence classification.
        
        The backbone outputs hidden states and the [CLS] token (first token)
        is passed through the score head to produce logits.
        """
        outputs = self.gfr(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Force output hidden states for [CLS]
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Compute the position of [SCORE] for each sequence
        # If pad to the right, sum of attention_mask gives the unpadded length; [SCORE] is at length - 1
        # score_positions = attention_mask.sum(dim=1) - 1  # Shape: (batch_size,) 

        # Ensure positions are within bounds
        # score_positions = torch.clamp(score_positions, min=0, max=hidden_states.size(1) - 1)

        # If pad to the left, the position of [SCORE] is at the last position of the sequence
        score_positions = hidden_states.size(1) - 1

        # Extract the hidden state of [SCORE] for each sequence in the batch
        batch_indices = torch.arange(hidden_states.size(0))  # [0, 1, 2, ..., batch_size-1]
        score_hidden = hidden_states[batch_indices, score_positions]  # Shape: (batch_size, hidden_size)

        # Pass the [SCORE] hidden state through a scoring head
        logits = self.score_head(score_hidden).squeeze(-1)  # Shape: (batch_size,)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            return (loss, logits)
        else:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
                "past_key_values": outputs.past_key_values
            }
        
    def prepare_input(self, documents: list, queries: list, tokenizer, max_length: int = 1024):
        """
        Prepares batched inputs for the model by truncating document tokens if needed and dynamically
        padding to the longest sequence in the batch rather than a fixed max_length.
        """
        input_ids_list = []
        token_type_ids_list = []
        
        # First, build tokenized sequences without padding.
        for document, query in zip(documents, queries):
            # Encode document and query without adding special tokens.
            doc_ids = tokenizer.encode(document, add_special_tokens=False)
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            score_id = tokenizer.convert_tokens_to_ids("[SCORE]")
            
            # Calculate available tokens for the document.
            # Reserve tokens for [SEP] and [SCORE] plus the query tokens.
            reserved_tokens = 2 + len(query_ids)
            available_doc_length = max_length - reserved_tokens
            
            if available_doc_length < 0:
                raise ValueError("max_length is too small to accommodate the query and required special tokens.")
            
            # Truncate document tokens if necessary.
            truncated_doc_ids = doc_ids[:available_doc_length]
            
            # Build the final input sequence: truncated_doc_ids + [SEP] + query_ids + [SCORE].
            input_ids = truncated_doc_ids + [tokenizer.sep_token_id] + query_ids + [score_id]
            input_ids_list.append(input_ids)
            
            # Create token type IDs:
            # Token type 0 for document tokens and [SEP], and token type 1 for query tokens and [SCORE].
            token_type_ids = [0] * (len(truncated_doc_ids) + 1) + [1] * (len(query_ids) + 1)
            token_type_ids_list.append(token_type_ids)
        
        # Determine the maximum sequence length in the batch.
        batch_max_length = max(len(seq) for seq in input_ids_list)
        
        # Pad sequences to the batch maximum length.
        padded_input_ids_list = [
            [tokenizer.pad_token_id] * (batch_max_length - len(seq)) + seq
            for seq in input_ids_list
        ]
        padded_token_type_ids_list = [
            [0] * (batch_max_length - len(seq)) + seq
            for seq in token_type_ids_list
        ]
        attention_masks_list = [
            [0] * (batch_max_length - len(seq)) + [1] * len(seq)
            for seq in input_ids_list
        ]
        
        # Convert lists to tensors.
        input_ids_tensor = torch.tensor(padded_input_ids_list, dtype=torch.long)
        token_type_ids_tensor = torch.tensor(padded_token_type_ids_list, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks_list, dtype=torch.long)
        
        return input_ids_tensor, token_type_ids_tensor, attention_mask_tensor

__all__ = ["GFRPreTrainedModel", "GFRModel", "GFRForCausalLM", "GFRModelWithTokenTypes", "GFRForSequenceScoring"]