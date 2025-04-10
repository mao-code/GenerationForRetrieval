from transformers.configuration_utils import PretrainedConfig


class GFR2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GFR2Model`]. It is used to instantiate a
    GFR2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GFR2 model.

    [Zyphra/GFR2-2.7B](https://huggingface.co/Zyphra/GFR2-2.7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the GFR2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GFR2Model`]
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 54):
            Number of hidden layers in the model.
        layers_block_type (`list`, *optional*):
            List of layer types, which can be either "mamba" or "hybrid".
        mamba_d_state (`int`, *optional*, defaults to 64): shape of the state space latents.
        mamba_d_conv (`int`, *optional*, defaults to 4): Size of the convolution kernel.
        mamba_expand (`int`, *optional*, defaults to 2): Expanding factor used to determine the intermediate size.
        mamba_ngroups (`int`, *optional*, defaults to 1):
            Number of groups for the evolution matrices of mamba 2.
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        time_step_limit (`tuple`, *optional*):
            Accepted range of time step values.
        n_mamba_heads (`int`, *optional*, defaults to 8):
            Number of heads for the evolution matrices of mamba 2.
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        chunk_size (`int`, *optional*, defaults to 256):
            Size of the chunks that will comprise the sequence.
        use_mem_eff_path (`bool`, *optional*, defaults to `False`):
            Whether or not to use the fused conv1d and scan in mamba2 layers.
        add_bias_linear (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in various layers
        intermediate_size (`int`, *optional*, defaults to 4 * hidden_size):
            Dimension of the MLP representations.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the MLP.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=None`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_mem_blocks (`int`, *optional*, defaults to 1):
            Number of unshared transformer blocks.
        use_shared_attention_adapter (`bool`, *optional*, defaults to `False`):
            If True, unshared adapters (formally the same as LoRA but used in the base model) will be added to the q, k, v projectors in the shared attention layers.
        adapter_rank (`int`, *optional*, defaults to 128):
            Rank of the adapter in the shared MLP and shared attention layers.
        use_mem_rope (`bool`, *optional*, defaults to `False`):
            If True, includes RoPE in the shared attention layers.
        rope_theta (`float`, *optional*, defaults to `10000.0`):
            The base period of the RoPE embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
            logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
            sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
            significantly.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        use_long_context (`bool`, *optional*, defaults to `False`):
            Activates the context-extended version of GFR by modifying RoPE.
    ```python
    >>> from transformers import GFR2Model, GFR2Config
    >>> # Initializing a GFR2-2.7B style configuration
    >>> configuration = GFR2Config()
    >>> # Initializing a model from the GFR2-2.7B style configuration
    >>> model = GFR2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "GFR2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_hidden_blocks=3,        # N GFR2Blocks,
        num_layers_per_block=8,     # 8 layers per block
        vocab_size=32000,
        max_position_embeddings=1024,
        hidden_size=1024,
        mamba_d_state=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_ngroups=1,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        time_step_limit=None,
        n_mamba_heads=8,
        use_conv_bias=True,
        chunk_size=128,
        use_mem_eff_path=True,
        add_bias_linear=False,
        intermediate_size=None,
        hidden_act="gelu",
        num_attention_heads=16,
        num_key_value_heads=None,
        attention_dropout=0.0,
        use_mem_rope=True,
        rope_theta=10000,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_long_context=False,

        # MLA configuration fields:
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        qk_head_dim=128,
        v_head_dim=128,
        attention_bias=False,
        # Optionally, define rope_scaling if using RoPE.
        rope_scaling=None,
        rope_interleave=True,

        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.num_hidden_blocks = num_hidden_blocks
        self.num_layers_per_block = num_layers_per_block

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        if intermediate_size is None:
            self.intermediate_size = 4 * hidden_size
        else:
            self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_blocks * num_layers_per_block
        self.num_attention_heads = num_attention_heads
        self.attention_hidden_size = 2 * hidden_size
        self.attention_head_dim = 2 * self.hidden_size // self.num_attention_heads
        self.attention_dropout = attention_dropout
        self.use_mem_rope = use_mem_rope
        self.use_long_context = use_long_context
        if use_mem_rope and use_long_context:
            a = 8
            rope_theta = rope_theta * a ** (self.attention_head_dim / (self.attention_head_dim - 2))
        self.rope_theta = rope_theta
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.add_bias_linear = add_bias_linear
        self.mamba_ngroups = mamba_ngroups
        self.n_mamba_heads = n_mamba_heads
        self.mamba_headdim = int(mamba_expand * hidden_size) // n_mamba_heads
        self.use_conv_bias = use_conv_bias
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        if use_long_context:
            self.max_position_embeddings = 16384
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.kv_channels = self.hidden_size // self.num_attention_heads
        self.num_query_groups = self.num_attention_heads

        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.use_mem_eff_path = use_mem_eff_path

        # MLA configuration fields:
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.rope_scaling = rope_scaling
        self.rope_interleave = rope_interleave
        self.attention_bias = attention_bias

__all__ = ["GFR2Config"]