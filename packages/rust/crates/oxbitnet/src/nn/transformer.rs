use std::sync::Arc;

use wgpu::BufferUsages;

use crate::gpu::buffer_pool::{BufferPool, GpuBuf};
use crate::gpu::pipeline::PipelineManager;
use crate::model::config::ModelConfig;
use crate::nn::attention::{Attention, KvCache};
use crate::nn::bg_cache::BgCache;
use crate::nn::bitlinear::{buf_entry, create_uniform_raw, create_uniform_u32_u32_f32};
use crate::nn::ffn::FFN;

const RMSNORM_WGSL: &str = include_str!("../shaders/rmsnorm.wgsl");
const ELEMENTWISE_WGSL: &str = include_str!("../shaders/elementwise.wgsl");

/// Single transformer block:
///   residual = x
///   x = attention(input_layernorm(x)) + residual
///   residual = x
///   x = ffn(post_attention_layernorm(x)) + residual
pub struct TransformerBlock {
    device: Arc<wgpu::Device>,
    config: ModelConfig,
    input_layer_norm: GpuBuf,
    post_attn_layer_norm: GpuBuf,
    pub(crate) attention: Attention,
    pub(crate) ffn: FFN,
    bg_cache: BgCache,
}

impl TransformerBlock {
    pub fn new(
        device: Arc<wgpu::Device>,
        config: ModelConfig,
        input_layer_norm: GpuBuf,
        post_attn_layer_norm: GpuBuf,
        attention: Attention,
        ffn: FFN,
    ) -> Self {
        Self {
            device,
            config,
            input_layer_norm,
            post_attn_layer_norm,
            attention,
            ffn,
            bg_cache: BgCache::new(),
        }
    }

    pub fn forward(
        &mut self,
        input: &GpuBuf,
        n: usize,
        kv_cache: &mut KvCache,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let hidden = self.config.hidden_size;

        // Self-attention with residual
        let normed_attn = self.dispatch_rmsnorm(encoder, input, &self.input_layer_norm.clone(), n, pipelines, pool);
        let attn_out = self.attention.forward(&normed_attn, n, kv_cache, encoder, pipelines, pool);

        let residual1 = self.dispatch_add(encoder, input, &attn_out, n * hidden, pipelines, pool);

        // FFN with residual
        let normed_ffn = self.dispatch_rmsnorm(encoder, &residual1, &self.post_attn_layer_norm.clone(), n, pipelines, pool);
        let ffn_out = self.ffn.forward(&normed_ffn, n, encoder, pipelines, pool);

        let output = self.dispatch_add(encoder, &residual1, &ffn_out, n * hidden, pipelines, pool);

        output
    }

    fn dispatch_rmsnorm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        n: usize,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let entry = pipelines.get_or_create_default("rmsnorm", RMSNORM_WGSL);
        let hidden = self.config.hidden_size;

        let output = pool.acquire(
            (n * hidden * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let params = create_uniform_u32_u32_f32(
            &self.device,
            n as u32,
            hidden as u32,
            self.config.rms_norm_eps,
        );

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transformer_rmsnorm"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, weight),
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

    fn dispatch_add(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        num_elements: usize,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let entry = pipelines.get_or_create_default("elementwise_0", ELEMENTWISE_WGSL);

        let output = pool.acquire(
            (num_elements * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let params_data = [
            (num_elements as u32).to_le_bytes(),
            0u32.to_le_bytes(), // add
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transformer_add"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, a),
                buf_entry(1, b),
                buf_entry(2, &output),
                buf_entry(3, &params),
            ],
        });

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(((num_elements + 255) / 256) as u32, 1, 1);

        output
    }

    pub fn clear_bg_cache(&mut self) {
        self.bg_cache.clear();
        self.attention.clear_bg_cache();
        self.ffn.clear_bg_cache();
    }
}
