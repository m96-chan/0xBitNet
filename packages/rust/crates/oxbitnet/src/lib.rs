//! # oxbitnet
//!
//! Run BitNet b1.58 ternary LLMs with wgpu.
//!
//! ```no_run
//! use oxbitnet::BitNet;
//! use futures::StreamExt;
//!
//! # async fn example() -> oxbitnet::Result<()> {
//! let mut bitnet = BitNet::load("model.gguf", Default::default()).await?;
//!
//! {
//!     let mut stream = bitnet.generate("Hello!", Default::default());
//!     while let Some(token) = stream.next().await {
//!         print!("{token}");
//!     }
//! }
//!
//! bitnet.dispose();
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod gpu;
pub mod model;
pub mod nn;
pub mod sampling;
pub mod tokenizer;

pub use error::{BitNetError, Result};
pub use model::config::{Activation, ModelConfig};
pub use model::loader::{LoadOptions, LoadProgress};
pub use tokenizer::{ChatMessage, Tokenizer};

use std::pin::Pin;
use std::sync::Arc;

use async_stream::stream;
use futures::Stream;
use tracing::debug;

use gpu::device::init_gpu;
use model::loader::load_model;
use nn::model::BitNetModel;

/// Generation options.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_k: 50,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }
}

/// High-level API for BitNet inference on wgpu.
pub struct BitNet {
    model: BitNetModel,
    tokenizer: Tokenizer,
}

impl BitNet {
    /// Load a BitNet model from a URL or local path.
    pub async fn load(source: &str, options: LoadOptions) -> Result<Self> {
        let gpu = init_gpu().await?;

        let result = load_model(
            source,
            Arc::clone(&gpu.device),
            Arc::clone(&gpu.queue),
            options,
        )
        .await?;

        let model = BitNetModel::build(
            Arc::clone(&gpu.device),
            Arc::clone(&gpu.queue),
            result.config,
            &result.weights,
            4096,
        )?;

        let tokenizer = if let Some(ref metadata) = result.metadata {
            Tokenizer::from_gguf_metadata(metadata)?
        } else {
            return Err(BitNetError::Tokenizer(
                "No tokenizer metadata found".into(),
            ));
        };

        Ok(Self { model, tokenizer })
    }

    /// Generate text from a prompt. Returns a stream of token strings.
    pub fn generate(
        &mut self,
        prompt: &str,
        options: GenerateOptions,
    ) -> Pin<Box<dyn Stream<Item = String> + '_>> {
        let input_ids = self.tokenizer.encode(prompt, true).unwrap_or_default();
        self.generate_from_ids(input_ids, options)
    }

    /// Generate text from chat messages. Returns a stream of token strings.
    pub fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        options: GenerateOptions,
    ) -> Pin<Box<dyn Stream<Item = String> + '_>> {
        let input_ids = self
            .tokenizer
            .apply_chat_template(messages)
            .unwrap_or_default();
        self.generate_from_ids(input_ids, options)
    }

    fn generate_from_ids(
        &mut self,
        input_ids: Vec<u32>,
        options: GenerateOptions,
    ) -> Pin<Box<dyn Stream<Item = String> + '_>> {
        let max_tokens = options.max_tokens;
        let temperature = options.temperature;
        let top_k = options.top_k;
        let repeat_penalty = options.repeat_penalty;
        let repeat_last_n = options.repeat_last_n;

        let eos_id = self.tokenizer.eos_token_id();
        let eot_id = self.tokenizer.eot_token_id();
        let im_end_id = self.tokenizer.im_end_token_id();

        Box::pin(stream! {
            self.model.reset_kv_cache();

            debug!("generate: {} input tokens", input_ids.len());

            // Prefill
            let mut logits = self.model.forward(&input_ids);
            let mut recent_tokens: Vec<u32> = Vec::new();

            for _ in 0..max_tokens {
                let mut logits_data = match self.model.read_logits(&logits).await {
                    Ok(data) => data,
                    Err(_) => break,
                };

                let window = if repeat_last_n > 0 {
                    let start = recent_tokens.len().saturating_sub(repeat_last_n);
                    &recent_tokens[start..]
                } else {
                    &recent_tokens
                };

                let next_token = sampling::sample_token(
                    &mut logits_data,
                    temperature,
                    top_k,
                    repeat_penalty,
                    window,
                );

                if next_token == eos_id || eot_id == Some(next_token) || im_end_id == Some(next_token) {
                    break;
                }

                recent_tokens.push(next_token);

                if let Ok(token_str) = self.tokenizer.decode_one(next_token) {
                    yield token_str;
                }

                // Decode step
                logits = self.model.forward(&[next_token]);
            }
        })
    }

    /// Release all GPU resources.
    pub fn dispose(&mut self) {
        self.model.reset_kv_cache();
    }
}
