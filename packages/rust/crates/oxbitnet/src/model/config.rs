/// Model architecture configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub activation: Activation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu2,
    Silu,
    Swiglu,
}

impl ModelConfig {
    /// Head dimension = hidden_size / num_attention_heads
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA group size = num_attention_heads / num_key_value_heads
    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Default config for microsoft/bitnet-b1.58-2B-4T
pub fn bitnet_2b_4t_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 128256,
        hidden_size: 2560,
        intermediate_size: 6912,
        num_hidden_layers: 30,
        num_attention_heads: 20,
        num_key_value_heads: 5,
        max_position_embeddings: 4096,
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        tie_word_embeddings: true,
        activation: Activation::Relu2,
    }
}

/// Default config for 1bitLLM/bitnet_b1_58-large (0.7B)
pub fn bitnet_0_7b_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32002,
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: 24,
        num_attention_heads: 32,
        num_key_value_heads: 32,
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        activation: Activation::Silu,
    }
}
