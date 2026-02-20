use std::sync::Arc;

use wgpu::BufferUsages;

use crate::gpu::buffer_pool::{BufferPool, GpuBuf};
use crate::gpu::pipeline::PipelineManager;
use crate::model::config::ModelConfig;
use crate::nn::bg_cache::BgCache;
use crate::nn::bitlinear::{buf_entry, create_uniform_raw, BitLinear};

const ROPE_WGSL: &str = include_str!("../shaders/rope.wgsl");
const SOFTMAX_WGSL: &str = include_str!("../shaders/softmax.wgsl");
const ATTENTION_WGSL: &str = include_str!("../shaders/attention.wgsl");

/// KV cache for autoregressive generation.
pub struct KvCache {
    pub key: wgpu::Buffer,
    pub value: wgpu::Buffer,
    pub seq_len: usize,
    pub max_seq_len: usize,
}

pub fn create_kv_cache(
    device: &wgpu::Device,
    config: &ModelConfig,
    max_seq_len: usize,
) -> KvCache {
    let kv_size =
        (max_seq_len * config.num_key_value_heads * config.head_dim() * 4) as u64;
    let key = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kv_key"),
        size: kv_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let value = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("kv_value"),
        size: kv_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    KvCache {
        key,
        value,
        seq_len: 0,
        max_seq_len,
    }
}

/// Multi-Head Attention with GQA support.
pub struct Attention {
    device: Arc<wgpu::Device>,
    config: ModelConfig,
    h_dim: usize,
    pub(crate) q_proj: BitLinear,
    pub(crate) k_proj: BitLinear,
    pub(crate) v_proj: BitLinear,
    pub(crate) o_proj: BitLinear,
    bg_cache: BgCache,
}

impl Attention {
    pub fn new(
        device: Arc<wgpu::Device>,
        config: ModelConfig,
        q_proj: BitLinear,
        k_proj: BitLinear,
        v_proj: BitLinear,
        o_proj: BitLinear,
    ) -> Self {
        let h_dim = config.head_dim();
        Self {
            device,
            config,
            h_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            bg_cache: BgCache::new(),
        }
    }

    /// Forward pass: input [N, hidden] → output [N, hidden]
    pub fn forward(
        &mut self,
        input: &GpuBuf,
        n: usize,
        kv_cache: &mut KvCache,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        // Q/K/V projections
        let q_buf = self.q_proj.forward(input, n, encoder, pipelines, pool);
        let k_buf = self.k_proj.forward(input, n, encoder, pipelines, pool);
        let v_buf = self.v_proj.forward(input, n, encoder, pipelines, pool);

        // RoPE
        let q_roped = self.apply_rope(encoder, &q_buf, n, num_heads, kv_cache.seq_len, pipelines, pool);
        let k_roped = self.apply_rope(encoder, &k_buf, n, num_kv_heads, kv_cache.seq_len, pipelines, pool);

        // Update KV cache
        self.append_to_cache(encoder, &k_roped, &v_buf, kv_cache, n);

        let total_seq = kv_cache.seq_len + n;

        // Attention scores: Q @ K^T * scale
        let scores = self.compute_scores(encoder, &q_roped, &kv_cache.key, n, total_seq, pipelines, pool);

        // Softmax
        let attn_weights = self.apply_softmax(encoder, &scores, num_heads * n, total_seq, pipelines, pool);

        // Attention output: weights @ V
        let attn_output = self.compute_attn_v(encoder, &attn_weights, &kv_cache.value, n, total_seq, pipelines, pool);

        // Output projection
        let output = self.o_proj.forward(&attn_output, n, encoder, pipelines, pool);

        output
    }

    fn apply_rope(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        n: usize,
        num_heads: usize,
        pos_offset: usize,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let entry = pipelines.get_or_create_default("rope", ROPE_WGSL);

        let output_size = (n * num_heads * self.h_dim * 4) as u64;
        let output = pool.acquire(output_size, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

        let params_data = [
            (n as u32).to_le_bytes(),
            (num_heads as u32).to_le_bytes(),
            (self.h_dim as u32).to_le_bytes(),
            (pos_offset as u32).to_le_bytes(),
            self.config.rope_theta.to_le_bytes(),
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, &output),
                buf_entry(2, &params),
            ],
        });

        let total_pairs = (n * num_heads * (self.h_dim / 2)) as u32;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((total_pairs + 255) / 256, 1, 1);

        output
    }

    fn append_to_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        cache: &KvCache,
        n: usize,
    ) {
        let kv_size = (n * self.config.num_key_value_heads * self.h_dim * 4) as u64;
        let offset = (cache.seq_len * self.config.num_key_value_heads * self.h_dim * 4) as u64;

        encoder.copy_buffer_to_buffer(k, 0, &cache.key, offset, kv_size);
        encoder.copy_buffer_to_buffer(v, 0, &cache.value, offset, kv_size);
    }

    fn compute_scores(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        n: usize,
        s: usize,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let entry = pipelines.get_or_create("attention_scores", ATTENTION_WGSL, "compute_scores", None);
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        let scores = pool.acquire(
            (num_heads * n * s * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let scale = 1.0 / (self.h_dim as f32).sqrt();
        let params_data = [
            (n as u32).to_le_bytes(),
            (s as u32).to_le_bytes(),
            (num_heads as u32).to_le_bytes(),
            (num_kv_heads as u32).to_le_bytes(),
            (self.h_dim as u32).to_le_bytes(),
            scale.to_le_bytes(),
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_scores"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, q),
                buf_entry(1, k),
                buf_entry(2, &scores),
                buf_entry(3, &params),
            ],
        });

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(
            ((n + 15) / 16) as u32,
            ((s + 15) / 16) as u32,
            num_heads as u32,
        );

        scores
    }

    fn apply_softmax(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        n: usize,
        d: usize,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let entry = pipelines.get_or_create_default("softmax", SOFTMAX_WGSL);

        let output = pool.acquire(
            (n * d * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let params_data = [
            (n as u32).to_le_bytes(),
            (d as u32).to_le_bytes(),
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, input),
                buf_entry(1, &output),
                buf_entry(2, &params),
            ],
        });

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);

        output
    }

    fn compute_attn_v(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        attn: &wgpu::Buffer,
        v: &wgpu::Buffer,
        n: usize,
        s: usize,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let entry = pipelines.get_or_create("attn_v", ATTENTION_WGSL, "attn_v", None);
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        let output_size = (n * num_heads * self.h_dim * 4) as u64;
        let output = pool.acquire(output_size, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

        let params_data = [
            (n as u32).to_le_bytes(),
            (s as u32).to_le_bytes(),
            (num_heads as u32).to_le_bytes(),
            (num_kv_heads as u32).to_le_bytes(),
            (self.h_dim as u32).to_le_bytes(),
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_v"),
            layout: &entry.bind_group_layout,
            entries: &[
                buf_entry(0, attn),
                buf_entry(1, v),
                buf_entry(2, &output),
                buf_entry(3, &params),
            ],
        });

        let total = (n * num_heads * self.h_dim) as u32;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((total + 255) / 256, 1, 1);

        output
    }

    pub fn clear_bg_cache(&mut self) {
        self.bg_cache.clear();
        self.q_proj.clear_bg_cache();
        self.k_proj.clear_bg_cache();
        self.v_proj.clear_bg_cache();
        self.o_proj.clear_bg_cache();
    }
}
