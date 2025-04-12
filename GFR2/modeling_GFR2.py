import math
import re
from itertools import cycle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from GFR2.configuration_GFR2 import GFR2Config

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

if is_mamba_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    selective_state_update, mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None


logger = logging.get_logger(__name__)


# _CONFIG_FOR_DOC = "Zyphra/GFR2-2.7B"


class GFR2RMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        *prefix_dims, last_dim = hidden_states.shape
        group_count = last_dim // self.group_size
        hidden_states_group = hidden_states.view(*prefix_dims, group_count, self.group_size)
        variance = hidden_states_group.pow(2).mean(-1, keepdim=True)
        hidden_states_group = hidden_states_group * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states_group.view(*prefix_dims, group_count * self.group_size)
        return self.weight * hidden_states.to(input_dtype)


class GFR2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GFR2RMSNorm is equivalent to T5LayerNorm
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


class GFR2HybridDynamicCache(DynamicCache):
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

    def __init__(
        self, config: GFR2Config, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.dtype = dtype
        self.has_previous_state = False
        self.intermediate_size = int(config.mamba_expand * config.hidden_size)
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.n_mamba_heads = config.n_mamba_heads
        self.transformer_layers = []
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.conv_states = {}
        self.ssm_states = {}

        # Loop over all hidden layers.
        for i in range(config.num_hidden_layers):
            self.conv_states[i] = torch.zeros(
                batch_size,
                self.intermediate_size + 2 * config.mamba_ngroups * config.mamba_d_state,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
            self.ssm_states[i] = torch.zeros(
                batch_size, self.n_mamba_heads, config.mamba_headdim, self.ssm_state_size, device=device, dtype=dtype
            )

            # Mark transformer layers based on the block structure.
            # In our model, transformer layers are at positions 0 and 4 in each block.
            if i % config.num_layers_per_block in [0, 4]:
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

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

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx].numel() == 0:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("GFR2HybridDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("GFR2HybridDynamicCache does not have a legacy cache equivalent.")

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()

class GFR2MLARotaryEmbedding(nn.Module):
    def __init__(self, config: GFR2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.to(device=x.device, dtype=torch.float) @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

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
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""
    TODO let's just use the original freqcis computation to not have the view
    transpose + reshape! This is not optimized!
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

class GFR2MultiHeadLatentAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GFR2Config, layer_idx: int, concat_input: Optional[bool] = False):
        super().__init__()
        self.config = config
        self.dim_scale = (2 if concat_input else 1)
        self.attn_hidden_size = config.hidden_size * self.dim_scale

        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta

        self.qk_rope_head_dim = config.qk_rope_head_dim  # Fixed at 64
        self.qk_head_dim = self.dim_scale * config.qk_head_dim  # 128 or 256 based on dim_scale
        self.qk_nope_head_dim = self.qk_head_dim - self.qk_rope_head_dim  # Adjust accordingly

        self.q_lora_rank = self.dim_scale * config.q_lora_rank
        self.kv_lora_rank = self.dim_scale * config.kv_lora_rank
        self.v_head_dim = self.dim_scale * config.v_head_dim

        self.is_causal = True
        self.q_a_proj = nn.Linear(self.attn_hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = GFR2RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.attn_hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = GFR2RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.attn_hidden_size,
            bias=config.attention_bias,
        )

        # self.scaling = self.qk_head_dim ** (-0.5)
        self.scaling = (self.qk_head_dim / self.dim_scale) ** -0.5

        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

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

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# Helper methods for segment sum computation

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


is_fast_path_available = all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))


class GFR2MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: GFR2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = int(config.mamba_expand * self.hidden_size)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = "silu"
        self.act = nn.SiLU()
        self.use_mem_eff_path = config.use_mem_eff_path

        self.n_groups = config.mamba_ngroups
        self.head_dim = config.mamba_headdim
        self.num_heads = self.config.n_mamba_heads
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=True,
            kernel_size=config.mamba_d_conv,
            groups=self.conv_dim,
            padding=config.mamba_d_conv - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.add_bias_linear,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = GFR2RMSNormGated(
            self.intermediate_size, group_size=self.intermediate_size // self.n_groups, eps=1e-5
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.add_bias_linear)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[GFR2HybridDynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # set up dimensions for reshapes later

        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

        # getting projected states from cache if it exists
        if cache_params is not None and cache_params.has_previous_state:
            in_projected_states = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
            d_mlp = (in_projected_states.shape[-1] - d_to_remove) // 2
            split_projection_dim = [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads]
            _, _, gate, hidden_states_B_C, dt = torch.split(in_projected_states, split_projection_dim, dim=-1)

            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )
            A = -torch.exp(self.A_log.float())  # (nheads,)

            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
        # if no cache is found, calling the kernel
        else:
            if attention_mask is not None and not torch.all(attention_mask == 1):
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                dtype = hidden_states.dtype
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
            # 1. Gated MLP's linear projection
            projected_states = self.in_proj(hidden_states)
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}
            if attention_mask is not None:
                input_not_masked = torch.all(attention_mask == 1)
            else:
                input_not_masked = True

            if self.use_mem_eff_path and self.training and cache_params is None and input_not_masked:
                out, ssm_state = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=True,
                    **dt_limit_kwargs,
                )

            else:
                gate, hidden_states_B_C, time_step = torch.split(
                    projected_states,
                    [self.intermediate_size, self.conv_dim, self.num_heads],
                    dim=-1,
                )

                # 1D Convolution
                if cache_params is not None:
                    hidden_states_B_C_t = hidden_states_B_C.transpose(1, 2)
                    conv_state = nn.functional.pad(
                        hidden_states_B_C_t, (self.conv_kernel_size - hidden_states_B_C_t.shape[-1], 0)
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_state)
                if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
                    )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)[:, :seq_len]
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )
                if attention_mask is not None and not torch.all(attention_mask == 1):
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    dtype = hidden_states.dtype
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    time_step,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output = self.norm(scan_output, gate)
                out = self.out_proj(scan_output)
        return out

    # fmt: off
    def torch_forward(self, input_states, cache_params: Optional[GFR2HybridDynamicCache]=None, attention_mask: Optional[torch.Tensor]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # Gated MLP's linear projection
        if cache_params is not None and cache_params.has_previous_state:
            projected_states =  self.in_proj(input_states.squeeze(1))
        else:
            if attention_mask is not None and not torch.all(attention_mask==1):
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    input_states = (input_states * attention_mask[:, :, None]).to(dtype)
            projected_states =  self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size -  2 * self.n_groups * self.ssm_state_size- self.num_heads) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
                [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1
        )

        # Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            if cache_params.has_previous_state:
                gate = gate.unsqueeze(1)
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                # handle batched generation - states are copied through
                conv_state[:, :, -1] = hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state.to(projected_states.device) * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]         # [batch, 1, intermediate_size] : decoding
            else:
                hidden_states = hidden_states.transpose(1,2)
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states).transpose(1,2))[:, :seq_len, :]     # [batch, intermediate_size, seq_len]
                if attention_mask is not None and not torch.all(attention_mask==1):
                    dtype = hidden_states.dtype
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        else:
            ssm_state = torch.zeros(
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_states, B, C = torch.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], dim=-1)
        A = -torch.exp(self.A_log.float())                            # [num_heads]
        if cache_params is not None and cache_params.has_previous_state:
            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_min) #, self.time_step_max)
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = torch.exp(dt[..., None] * A)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = dB * hidden_states[..., None]

            # State calculation
            cache_params.ssm_states[self.layer_idx].copy_(
                cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_min)
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len,  -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]


            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(segment_sum(A))

            # First, contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, : ,:]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)


            # Step 2: Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Step 3: Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

            # (right term of low-rank factorization of off-diagonal blocks; B terms)

            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
            # permute back B * decay states
            states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]  * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)
            if cache_params is not None and cache_params.has_previous_state:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...]
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

            states_permuted = states.permute(0, 2, 1, 3, 4)
            result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
            new_states = result.permute(0, 2, 1, 3, 4)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            # compute Yoff
            C_times_states = (C[..., None, :] * states[:, :, None, ...])
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])
            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)

            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[GFR2HybridDynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

        return self.torch_forward(hidden_states, cache_params, attention_mask)

class GFR2MLP(nn.Module):
    def __init__(self, config, concat_input: Optional[bool] = False):
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

class GFR2AttentionDecoderLayer(nn.Module):
    def __init__(self, config: GFR2Config, layer_idx: Optional[int] = None, concat_input: Optional[bool] = False):
        super().__init__()

        self.layer_idx = layer_idx
        attn_hidden_size = config.hidden_size * (2 if concat_input else 1)

        self.input_layernorm = GFR2RMSNorm(attn_hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GFR2MultiHeadLatentAttention(config, layer_idx=layer_idx, concat_input=concat_input)

        self.pre_ff_layernorm = GFR2RMSNorm(attn_hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = GFR2MLP(config, concat_input=concat_input)

        self.concat_input = concat_input

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: Optional[torch.Tensor]=None,
        layer_idx: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[GFR2HybridDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[torch.LongTensor] = None,
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
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`GFR2HybridDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """
        if self.concat_input and original_hidden_states is not None:
            # Concatenate the original hidden states with the current hidden states along the last dimension
            # Therefore, the shape of `hidden_states` will be (batch_size, seq_len, 2 * hidden_size)
            hidden_states = torch.concatenate([hidden_states, original_hidden_states], dim=-1)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            layer_idx=self.layer_idx,
            attention_mask=causal_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            **kwargs,
        )

        # residual connection after attention
        hidden_states = residual + hidden_states

        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

class GFR2MambaDecoderLayer(nn.Module):
    def __init__(self, config: GFR2Config, layer_idx: int):
        super().__init__()
        self.mamba = GFR2MambaMixer(config=config, layer_idx=layer_idx)
        self.input_layernorm = GFR2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_ff_layernorm = GFR2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = GFR2MLP(config)

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[GFR2HybridDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        transformer_hidden_states: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`GFR2HybridDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """
        residual = hidden_states

        # `transformer_hidden_states` is the output from shared transformer + linear layer
        # `transformer_hidden_states` is then added to the input to the mamba layer below
        hidden_states = (
            hidden_states + transformer_hidden_states if transformer_hidden_states is not None else hidden_states
        )

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

class GFR2HybridLayer(nn.Module):
    def __init__(
        self, 
        transformer: GFR2AttentionDecoderLayer, 
        linear: nn.Linear, 
        mamba: GFR2MambaDecoderLayer
    ):
        super().__init__()
        self.linear = linear
        self.mamba_decoder = mamba
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[GFR2HybridDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        layer_outputs = self.transformer(
            hidden_states,
            original_hidden_states=original_hidden_states,
            layer_idx=layer_idx,
            attention_mask=causal_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        transformer_hidden_states = layer_outputs[0]

        if output_attentions:
            self_attn_weights = layer_outputs[1]

        transformer_hidden_states = self.linear(transformer_hidden_states)

        layer_outputs = self.mamba_decoder(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            cache_position=cache_position,
            transformer_hidden_states=transformer_hidden_states,
        )

        if output_attentions:
            layer_outputs = (layer_outputs[0], self_attn_weights) + layer_outputs[2:]

        return layer_outputs

class GFR2Block(nn.Module):
    def __init__(self, config, base_idx: int):
        """
        Each block consists of:
         - Pre-hybrid processing: one attention layer followed by three Mamba layers.
         - A hybrid layer that internally performs concatenation with the original hidden state,
           applies a transformer (with concat_input), a linear projection, and a Mamba step (Mamba4).
        """
        super().__init__()
        # Pre-hybrid layers
        self.attention = GFR2AttentionDecoderLayer(config, layer_idx=base_idx+0, concat_input=False)
        self.mamba1 = GFR2MambaDecoderLayer(config, layer_idx=base_idx+1)
        self.mamba2 = GFR2MambaDecoderLayer(config, layer_idx=base_idx+2)
        self.mamba3 = GFR2MambaDecoderLayer(config, layer_idx=base_idx+3)
        
        # Hybrid layer that combines steps 7-11 (and includes the Mamba block internally)
        self.hybrid = GFR2HybridLayer(
            transformer=GFR2AttentionDecoderLayer(config, layer_idx=base_idx+4, concat_input=True),
            linear=nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True),
            mamba=GFR2MambaDecoderLayer(config, layer_idx=base_idx+5)
        )

        self.mamba5 = GFR2MambaDecoderLayer(config, layer_idx=base_idx+6)
        self.mamba6 = GFR2MambaDecoderLayer(config, layer_idx=base_idx+7)

    def forward(self,
        hidden_states: torch.Tensor,
        original_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[GFR2HybridDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        block_attns = [] if output_attentions else None

        residual = hidden_states
        attn_out = self.attention(
            hidden_states,
            attention_mask=causal_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
        hidden_states = residual + attn_out[0]
        if output_attentions:
            block_attns.append(attn_out[1])
        
        # Mamba layers 1-3.
        for mamba_layer in [self.mamba1, self.mamba2, self.mamba3]:
            mamba_out = mamba_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )
            hidden_states = mamba_out[0]
            if output_attentions:
                block_attns.append(mamba_out[1])
        
        # Hybrid processing:
        # This internally concatenates the current hidden_states with the original embeddings,
        # applies a transformer, projects back to hidden_size, adds a residual, and then runs a Mamba block.
        hybrid_out = self.hybrid(
            hidden_states,
            original_hidden_states=original_hidden_states,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
        hidden_states = hybrid_out[0]
        if output_attentions:
            block_attns.append(hybrid_out[1])

        # Mamba layers 5 and 6.
        for mamba_layer in [self.mamba5, self.mamba6]:
            mamba_out = mamba_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )
            hidden_states = mamba_out[0]
            if output_attentions:
                block_attns.append(mamba_out[1])
        
        if output_attentions:
            return (hidden_states, block_attns)
        
        return (hidden_states,)

class GFR2PreTrainedModel(PreTrainedModel):
    config_class = GFR2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GFR2AttentionDecoderLayer", "GFR2MambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_sdpa = True
    _supports_cache_class = True  # Note: only supports GFR2HybridDynamicCache
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
        elif isinstance(module, GFR2MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.n_mamba_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True

class GFR2Model(GFR2PreTrainedModel):
    def __init__(self, config: GFR2Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self._tied_weights_keys = ["embed_tokens.weight"]

        self.num_hidden_blocks = getattr(config, "num_hidden_blocks", 1)
        self.num_layers_per_block = getattr(config, "num_layers_per_block", 8) # not include the linear layer
        self.num_hidden_layers = config.num_hidden_layers  # total number of sub-layers

        self.layers = nn.ModuleList()
        # Build the model as a stack of blocks.
        # Here each block uses 8 internal layer indices (1 attn, 3 mamba, 1 attn + 1 linear(not included), 3 mamba).
        self.blocks = nn.ModuleList([
            GFR2Block(config, base_idx=block_idx * self.num_layers_per_block)
            for block_idx in range(self.num_hidden_blocks)
        ])

        # Final normalization layer.
        self.final_layernorm = GFR2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.score_head = nn.Linear(config.hidden_size, 1)

        self._attn_implementation = config._attn_implementation        
        if config.use_mem_rope:
            if config.use_long_context:
                logger.warning_once(
                    "`use_long_context` set to `True`: using rescaled `rope_theta` and extended `max_position_embeddings`."
                )
            self.rotary_emb = GFR2MLARotaryEmbedding(config)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[GFR2HybridDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        original_hidden_states = torch.clone(inputs_embeds)

        if use_cache and past_key_values is None:
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            past_key_values = GFR2HybridDynamicCache(self.config, batch_size, dtype=self.dtype, device=self.device)

        if cache_position is None:
            first_transformer_layer_id = 0
            past_seen_tokens = (
                past_key_values.get_seq_length(layer_idx=first_transformer_layer_id)
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # create position embeddings to be shared across the decoder layers
        if self.config.use_mem_rope:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        all_hidden_states = [] if output_hidden_states else None
        all_self_attns = [] if output_attentions else None

        # Process each hybrid block sequentially.
        for block_idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                block_out = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    original_hidden_states,
                    attention_mask,
                    causal_mask,
                    past_key_values,
                    output_attentions,
                    position_embeddings,
                    use_cache,
                    cache_position
                )
            else:
                block_out = block(
                    hidden_states,
                    original_hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs
                )

            hidden_states = block_out[0]

            if output_attentions:
                if block_out[1] is not None:
                    all_self_attns.extend(block_out[1])

        # Final normalization.
        hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

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

class GFR2ModelWithTokenTypes(GFR2Model):
    def __init__(self, config):
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
        past_key_values: Optional["GFR2HybridDynamicCache"] = None,
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


# Adapted from transformers.models.jamba.modeling_jamba.JambaForCausalLM with Jamba->GFR2, JAMBA->GFR2
class GFR2ForCausalLM(GFR2PreTrainedModel, GenerationMixin):
    def __init__(self, config: GFR2Config):
        super().__init__(config)
        self.model = GFR2Model(config)
        self._tied_weights_keys = ["lm_head.weight", *self.model._tied_weights_keys]
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[GFR2HybridDynamicCache] = None,
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training:
            use_cache = False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

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
        # Overwitten -- has a unique cache type, `GFR2HybridDynamicCache`

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
                or cache_position[-1] >= input_ids.shape[1]  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

            # Ensure past_key_values is GFR2HybridDynamicCache
            if not isinstance(past_key_values, GFR2HybridDynamicCache):
                if isinstance(past_key_values, DynamicCache):
                    new_cache = GFR2HybridDynamicCache(
                        self.config, input_ids.shape[0], dtype=self.dtype, device=self.device
                    )
                    for layer_idx in range(len(past_key_values.key_cache)):
                        new_cache.key_cache[layer_idx] = past_key_values.key_cache[layer_idx]
                        new_cache.value_cache[layer_idx] = past_key_values.value_cache[layer_idx]
                        # Copy conv_states and ssm_states if they exist in your DynamicCache (adjust as needed)
                        if hasattr(past_key_values, 'conv_states'):
                            new_cache.conv_states[layer_idx] = past_key_values.conv_states[layer_idx]
                        else:
                            print("Warning: conv_states not found in past_key_values.")
                        if hasattr(past_key_values, 'ssm_states'):
                            new_cache.ssm_states[layer_idx] = past_key_values.ssm_states[layer_idx]
                        else:
                            print("Warning: ssm_states not found in past_key_values.")
                    past_key_values = new_cache
                else:
                    raise ValueError(f"past_key_values must be GFR2HybridDynamicCache, got {type(past_key_values)}")
        else:
            past_key_values = GFR2HybridDynamicCache(
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

class GFR2ForSequenceClassification(GFR2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GFR2Model(config)
        self._tied_weights_keys = self.model._tied_weights_keys
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

class GFR2ForSequenceScoring(GFR2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.gfr2 = GFR2ModelWithTokenTypes(config)

        self.token_type_embedding = nn.Embedding(2, config.hidden_size)
        self.score_head = nn.Linear(config.hidden_size, 1, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.gfr2.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        return self.gfr2.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["GFR2HybridDynamicCache"] = None,
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
        outputs = self.gfr2(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

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
    
    def prepare_documents_input(self, documents: list, tokenizer):
        """
        Prepares batched document inputs.
        Each document is tokenized (with a trailing [SEP]) and then dynamically padded
        to the maximum document length within the batch.
        """

        # Append [SEP] to each document
        doc_sequences = [doc + " [SEP]" for doc in documents]
        inputs = tokenizer(
            doc_sequences,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ) # shape: (batch_size, doc_max_len)
        
        input_ids = inputs["input_ids"]
        token_type_ids = torch.zeros_like(input_ids)  # All 0 for doc_ids and [SEP]
        attention_mask = inputs["attention_mask"]

        return input_ids, token_type_ids, attention_mask

    # TODO: Should not pad on the query part because we pad to the left and need to get [SCORE] for the last token
    # TODO: Need to design a mechanism for caching usage
    def prepare_query_input(self, queries: list, tokenizer):
        """
        Prepares batched query inputs.
        """
        raise NotImplementedError("This method is not implemented yet.")
    
        # Append [SCORE] to each query
        query_sequences = [query + " [SCORE]" for query in queries]
        inputs = tokenizer(
            query_sequences,
            padding=False,
            truncation=False,
            # padding="longest",
            # truncation=True,
            return_tensors="pt"
        ) # shape: (batch_size, query_max_len)
        
        input_ids = inputs["input_ids"]
        token_type_ids = torch.ones_like(input_ids)  # All 1 for query_ids and [SCORE]
        attention_mask = inputs["attention_mask"]
        
        return input_ids, token_type_ids, attention_mask

__all__ = ["GFR2ForCausalLM", "GFR2ForSequenceClassification", "GFR2ForSequenceScoring", "GFR2Model", "GFR2ModelWithTokenTypes", "GFR2PreTrainedModel"]