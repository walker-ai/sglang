# Adapted from
# https://huggingface.co/THUDM/chatglm2-6b/blob/main/configuration_chatglm.py

from transformers.configuration_utils import PretrainedConfig


# what changes need to made to this class?
class GLMConfig(PretrainedConfig):
    model_type = "glm"

    # A dict that maps model specific attribute names
    # to the standardized naming of attributes.
    attribute_map = {
        "num_hidden_layers": "num_layers",
    }

    def __init__(
            self,
            num_layers=24,
            vocab_size=30592,
            hidden_size=1024,
            num_attention_heads=64,
            num_key_value_heads=-1,
            embedding_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            output_dropout_prob=0.1,
            max_sequence_length=4096,
            max_position_embeddings=-1,
            checkpoint_activations=False,
            checkpoint_num_layers=1,
            parallel_output=True,
            relative_encoding=False,
            block_position_encoding=True,
            output_predict=False,
            spell_length=None,
            spell_func="lstm",
            attention_scale=1.0,
            initializer_range=0.02,
            pool_token="cls",
            max_memory_length=0,
            bf16=True,
            intermediate_size=None,
            use_rotary=False,
            rope_scaling=1.0,
            use_cache=True,
            use_rmsnorm=False,
            use_swiglu=False,
            rotary_type=None,
            no_repeat_ngram_size=0,
            tie_word_embeddings=True,
            use_bias=True,
            use_qkv_bias=False,
            focused_attention=False,
            unidirectional=False,
            unidirectional_attention=False,
            rope_type=None,
            gate_up=False,
            mlp_version="v1",
            num_experts=0,
            moe_config=None,
            norm_head=False,
            rope_theta=10000,
            head_dim=0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # 是否是mqa或gqa
        self.num_key_value_heads = num_key_value_heads \
            if num_key_value_heads > 0 else self.num_attention_heads
        self.head_dim = head_dim if head_dim > 0 else \
            (hidden_size // num_attention_heads)
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.max_position_embeddings = max_position_embeddings \
            if max_position_embeddings > 0 else self.max_sequence_length
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.parallel_output = parallel_output
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        self.output_predict = output_predict
        self.spell_length = spell_length
        self.spell_func = spell_func
        self.attention_scale = attention_scale
        self.initializer_range = initializer_range
        self.pool_token = pool_token
        self.max_memory_length = max_memory_length
        self.bf16 = bf16
        self.intermediate_size = intermediate_size

        # rope的相关配置
        self.use_rotary = use_rotary
        self.rotary_type = rotary_type.lower() if rotary_type else None
        if self.rotary_type == 'none':
            self.rotary_type = None
        # 旋转位置类型，强制use_rotary为true
        if self.rotary_type:
            if self.rotary_type != '1d' and self.rotary_type != 'full-1d':
                raise ValueError(f"antglm `rotary_type` must be `1d` or `full-1d` rotary, got {self.rotary_type}")
            self.use_rotary = True
            self.block_position_encoding = False
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        if self.use_rotary:
            self._rope_type_validation()
        else:
            self.rope_type = None
        # 兼容命名，rope_scaling删去百灵的定义
        # self.rope_scaling = rope_scaling
        self.rope_scaling = self.rope_type

        self.use_cache = use_cache
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.tie_word_embeddings = tie_word_embeddings
        self.use_bias = use_bias
        self.use_qkv_bias = use_qkv_bias
        self.focused_attention = focused_attention
        self.causal_lm = unidirectional_attention or unidirectional
        self.gate_up = gate_up
        self.mlp_version = mlp_version
        if self.mlp_version == "v2":
            self.gate_up = True
        self.norm_head = norm_head

        self.num_experts = num_experts
        self.moe_config = moe_config
        if self.num_experts > 0 and self.moe_config:
            self.top_k = self.moe_config.get("top_k", 2)
            self.norm_expert_prob = self.moe_config.get("norm_expert_prob", True)
            # 共享专家版moe
            self.num_shared_experts = self.moe_config.get("num_shared_experts", 0)
            self.fine_grained_factor = self.moe_config.get("fine_grained_factor", 1)
            self.expert_intermediate_size = self.moe_config.get("expert_intermediate_size", None)
            if self.expert_intermediate_size is None:
                self.expert_intermediate_size = int(self.intermediate_size // self.fine_grained_factor)
        # 如果是双向模型，需要强制使用xformers
        self.enforce_xformers = not self.causal_lm

    def _rope_type_validation(self):
        """
        Validate the `rope_type` configuration.
        """
        if self.rope_type is None:
            return

        if not isinstance(self.rope_type, dict):
            raise ValueError(f"`rope_type` must be a dictionary, got {self.rope_type}")

        rope_scaling_type = self.rope_type.get("type", None)
        rope_scaling_factor = self.rope_type.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["dynamic", "yarn", "linear", "su"]:
            raise ValueError(
                f"`rope_type`'s name field must be one of ['dynamic', 'yarn', 'linear', 'su'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_type`'s factor field must be an float > 1, got {rope_scaling_factor}")
        if rope_scaling_type == "yarn" or rope_scaling_type == "su":
            original_max_position_embeddings = self.rope_type.get("original_max_position_embeddings", None)
            if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
                raise ValueError(
                    f"`rope_type.original_max_position_embeddings` must be set to an int when using su and yarn")
