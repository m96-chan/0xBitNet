use std::sync::Arc;

use wgpu::BufferUsages;

use crate::error::{BitNetError, Result};
use crate::gpu::buffer_pool::{BufferPool, GpuBuf};
use crate::gpu::pipeline::PipelineManager;
use crate::model::config::ModelConfig;
use crate::model::weights::WeightStore;
use crate::nn::attention::{create_kv_cache, Attention, KvCache};
use crate::nn::bitlinear::{buf_entry, create_uniform_raw, create_uniform_u32_u32_f32, BitLinear};
use crate::nn::ffn::FFN;
use crate::nn::transformer::TransformerBlock;

const EMBEDDING_WGSL: &str = include_str!("../shaders/embedding.wgsl");
const RMSNORM_WGSL: &str = include_str!("../shaders/rmsnorm.wgsl");
const F32_MATMUL_WGSL: &str = include_str!("../shaders/f32_matmul.wgsl");

/// Full BitNet model: embedding → N × transformer → final RMSNorm → LM head
pub struct BitNetModel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: PipelineManager,
    pub(crate) pool: BufferPool,
    pub config: ModelConfig,

    embed_tokens: GpuBuf,
    layers: Vec<TransformerBlock>,
    final_norm: GpuBuf,
    lm_head: LmHead,
    kv_caches: Vec<KvCache>,
}

enum LmHead {
    Tied,                    // Use embed_tokens
    Separate(BitLinear),
}

impl BitNetModel {
    /// Build a full model from loaded weights.
    pub fn build(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: ModelConfig,
        weights: &WeightStore,
        max_seq_len: usize,
    ) -> Result<Self> {
        let pipelines = PipelineManager::new(Arc::clone(&device));
        let pool = BufferPool::new(Arc::clone(&device), 256);

        let require = |name: &str| -> Result<GpuBuf> {
            weights
                .get(name)
                .cloned()
                .ok_or_else(|| BitNetError::MissingWeight(name.to_string()))
        };

        let embed_tokens = require("model.embed_tokens.weight")?;
        let final_norm = require("model.norm.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut kv_caches = Vec::with_capacity(config.num_hidden_layers);

        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let head_dim = config.head_dim();

            let input_ln = require(&format!("{p}.input_layernorm.weight"))?;
            let post_attn_ln = require(&format!("{p}.post_attention_layernorm.weight"))?;

            let attn_sub_norm = weights.get(&format!("{p}.self_attn.sub_norm.weight")).cloned();
            let ffn_sub_norm = weights.get(&format!("{p}.mlp.sub_norm.weight")).cloned();

            let q_proj = BitLinear::new(
                Arc::clone(&device),
                require(&format!("{p}.self_attn.q_proj.weight"))?,
                require(&format!("{p}.self_attn.q_proj.weight_scale"))?,
                None,
                config.hidden_size,
                config.num_attention_heads * head_dim,
            );
            let k_proj = BitLinear::new(
                Arc::clone(&device),
                require(&format!("{p}.self_attn.k_proj.weight"))?,
                require(&format!("{p}.self_attn.k_proj.weight_scale"))?,
                None,
                config.hidden_size,
                config.num_key_value_heads * head_dim,
            );
            let v_proj = BitLinear::new(
                Arc::clone(&device),
                require(&format!("{p}.self_attn.v_proj.weight"))?,
                require(&format!("{p}.self_attn.v_proj.weight_scale"))?,
                None,
                config.hidden_size,
                config.num_key_value_heads * head_dim,
            );
            let o_proj = BitLinear::new(
                Arc::clone(&device),
                require(&format!("{p}.self_attn.o_proj.weight"))?,
                require(&format!("{p}.self_attn.o_proj.weight_scale"))?,
                attn_sub_norm,
                config.num_attention_heads * head_dim,
                config.hidden_size,
            );

            let attention = Attention::new(
                Arc::clone(&device),
                config.clone(),
                q_proj, k_proj, v_proj, o_proj,
            );

            let up_proj = BitLinear::new(
                Arc::clone(&device),
                require(&format!("{p}.mlp.up_proj.weight"))?,
                require(&format!("{p}.mlp.up_proj.weight_scale"))?,
                None,
                config.hidden_size,
                config.intermediate_size,
            );
            let down_proj = BitLinear::new(
                Arc::clone(&device),
                require(&format!("{p}.mlp.down_proj.weight"))?,
                require(&format!("{p}.mlp.down_proj.weight_scale"))?,
                ffn_sub_norm,
                config.intermediate_size,
                config.hidden_size,
            );

            let gate_proj = if weights.has(&format!("{p}.mlp.gate_proj.weight")) {
                Some(BitLinear::new(
                    Arc::clone(&device),
                    require(&format!("{p}.mlp.gate_proj.weight"))?,
                    require(&format!("{p}.mlp.gate_proj.weight_scale"))?,
                    None,
                    config.hidden_size,
                    config.intermediate_size,
                ))
            } else {
                None
            };

            let ffn = FFN::new(
                Arc::clone(&device),
                config.clone(),
                up_proj, down_proj, gate_proj,
            );

            layers.push(TransformerBlock::new(
                Arc::clone(&device),
                config.clone(),
                input_ln,
                post_attn_ln,
                attention,
                ffn,
            ));

            kv_caches.push(create_kv_cache(&device, &config, max_seq_len));
        }

        let lm_head = if config.tie_word_embeddings || !weights.has("lm_head.weight") {
            LmHead::Tied
        } else {
            LmHead::Separate(BitLinear::new(
                Arc::clone(&device),
                require("lm_head.weight")?,
                require("lm_head.weight_scale")?,
                weights.get("lm_head.input_norm.weight").cloned().or_else(|| Some(final_norm.clone())),
                config.hidden_size,
                config.vocab_size,
            ))
        };

        Ok(Self {
            device,
            queue,
            pipelines,
            pool,
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            kv_caches,
        })
    }

    /// Forward pass: token IDs → logits buffer [1, vocab_size] f32
    pub fn forward(&mut self, token_ids: &[u32]) -> GpuBuf {
        let n = token_ids.len();
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Upload token IDs
        let token_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("token_ids"),
            size: (token_ids.len() * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = token_buffer.slice(..).get_mapped_range_mut();
            let bytes: &[u8] = bytemuck::cast_slice(token_ids);
            view[..bytes.len()].copy_from_slice(bytes);
        }
        token_buffer.unmap();
        let token_buffer = Arc::new(token_buffer);

        // Embedding lookup
        let mut hidden = self.dispatch_embedding(&mut encoder, &token_buffer, n);

        // Transformer layers
        for i in 0..self.layers.len() {
            let new_hidden = {
                let kv = &mut self.kv_caches[i];
                self.layers[i].forward(&hidden, n, kv, &mut encoder, &mut self.pipelines, &self.pool)
            };
            hidden = new_hidden;
            self.kv_caches[i].seq_len += n;
        }

        // Final RMSNorm
        let normed = self.dispatch_final_norm(&mut encoder, &hidden, n);

        // Extract last token for LM head
        let lm_input = if n > 1 {
            let lm_buf = self.pool.acquire(
                (self.config.hidden_size * 4) as u64,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            );
            encoder.copy_buffer_to_buffer(
                &normed,
                ((n - 1) * self.config.hidden_size * 4) as u64,
                &lm_buf,
                0,
                (self.config.hidden_size * 4) as u64,
            );
            lm_buf
        } else {
            normed
        };

        // LM head (always N=1)
        let logits = match &mut self.lm_head {
            LmHead::Separate(ref mut bl) => {
                bl.forward(&lm_input, 1, &mut encoder, &mut self.pipelines, &self.pool)
            }
            LmHead::Tied => {
                self.dispatch_lm_head(&mut encoder, &lm_input, 1)
            }
        };

        self.queue.submit(std::iter::once(encoder.finish()));
        logits
    }

    /// Read logits from GPU buffer to CPU.
    pub async fn read_logits(&self, logits: &GpuBuf) -> Result<Vec<f32>> {
        let size = self.config.vocab_size * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits_staging"),
            size: size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(logits, 0, &staging, 0, size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = tokio::sync::oneshot::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.await
            .map_err(|_| BitNetError::BufferMap)?
            .map_err(|_| BitNetError::BufferMap)?;

        let data = staging.slice(..).get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result = floats.to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    pub fn reset_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.seq_len = 0;
        }
        for layer in &mut self.layers {
            layer.clear_bg_cache();
        }
        if let LmHead::Separate(ref mut bl) = self.lm_head {
            bl.clear_bg_cache();
        }
    }

    fn dispatch_embedding(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        token_buffer: &wgpu::Buffer,
        n: usize,
    ) -> GpuBuf {
        let entry = self.pipelines.get_or_create_default("embedding", EMBEDDING_WGSL);

        let output_size = (n * self.config.hidden_size * 4) as u64;
        let output = self.pool.acquire(
            output_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let params_data = [
            (n as u32).to_le_bytes(),
            (self.config.hidden_size as u32).to_le_bytes(),
            (self.config.vocab_size as u32).to_le_bytes(),
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedding"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, token_buffer),
                buf_entry(1, &self.embed_tokens),
                buf_entry(2, &output),
                buf_entry(3, &params),
            ],
        });

        let total = (n * self.config.hidden_size) as u32;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);

        output
    }

    fn dispatch_final_norm(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        n: usize,
    ) -> GpuBuf {
        let entry = self.pipelines.get_or_create_default("rmsnorm", RMSNORM_WGSL);

        let output = self.pool.acquire(
            (n * self.config.hidden_size * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let params = create_uniform_u32_u32_f32(
            &self.device,
            n as u32,
            self.config.hidden_size as u32,
            self.config.rms_norm_eps,
        );

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("final_norm"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, &self.final_norm),
                buf_entry(2, &output),
                buf_entry(3, &params),
            ],
        });

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);

        output
    }

    fn dispatch_lm_head(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        n: usize,
    ) -> GpuBuf {
        let v = self.config.vocab_size;
        let d = self.config.hidden_size;
        let entry = self.pipelines.get_or_create_default("f32_matmul", F32_MATMUL_WGSL);

        let output = self.pool.acquire(
            (n * v * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let params_data = [
            (n as u32).to_le_bytes(),
            (v as u32).to_le_bytes(),
            (d as u32).to_le_bytes(),
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lm_head"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, &self.embed_tokens),
                buf_entry(2, &output),
                buf_entry(3, &params),
            ],
        });

        let total = (n * v) as u32;
        let wg_x = total.min(65535);
        let wg_y = total.div_ceil(65535);

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);

        output
    }
}
