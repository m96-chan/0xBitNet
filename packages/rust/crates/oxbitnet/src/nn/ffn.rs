use std::sync::Arc;

use wgpu::BufferUsages;

use crate::gpu::buffer_pool::{BufferPool, GpuBuf};
use crate::gpu::pipeline::PipelineManager;
use crate::model::config::{Activation, ModelConfig};
use crate::nn::bg_cache::BgCache;
use crate::nn::bitlinear::{buf_entry, create_uniform_raw, BitLinear};

const ACTIVATION_WGSL: &str = include_str!("../shaders/activation.wgsl");
const ELEMENTWISE_WGSL: &str = include_str!("../shaders/elementwise.wgsl");

/// Feed-Forward Network with gated activation.
///
/// Gated ReLU² (official 2B-4T): out = down_proj(relu²(gate(x)) * up(x))
/// SwiGLU (community models): out = down_proj(silu(gate(x)) * up(x))
pub struct FFN {
    device: Arc<wgpu::Device>,
    config: ModelConfig,
    pub(crate) up_proj: BitLinear,
    pub(crate) down_proj: BitLinear,
    pub(crate) gate_proj: Option<BitLinear>,
    bg_cache: BgCache,
}

impl FFN {
    pub fn new(
        device: Arc<wgpu::Device>,
        config: ModelConfig,
        up_proj: BitLinear,
        down_proj: BitLinear,
        gate_proj: Option<BitLinear>,
    ) -> Self {
        Self {
            device,
            config,
            up_proj,
            down_proj,
            gate_proj,
            bg_cache: BgCache::new(),
        }
    }

    pub fn forward(
        &mut self,
        input: &GpuBuf,
        n: usize,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        if self.gate_proj.is_some() {
            self.forward_gated(input, n, encoder, pipelines, pool)
        } else {
            self.forward_simple(input, n, encoder, pipelines, pool)
        }
    }

    fn forward_gated(
        &mut self,
        input: &GpuBuf,
        n: usize,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let act_type = if self.config.activation == Activation::Relu2 { 0u32 } else { 1u32 };

        let gate = self.gate_proj.as_mut().unwrap().forward(input, n, encoder, pipelines, pool);
        let up = self.up_proj.forward(input, n, encoder, pipelines, pool);

        let num_elements = n * self.config.intermediate_size;
        let gate_activated = self.apply_activation(encoder, &gate, num_elements, act_type, pipelines, pool);

        let gated = self.apply_elementwise(encoder, &gate_activated, &up, num_elements, 1, pipelines, pool);

        
        self.down_proj.forward(&gated, n, encoder, pipelines, pool)
    }

    fn forward_simple(
        &mut self,
        input: &GpuBuf,
        n: usize,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let act_type = if self.config.activation == Activation::Relu2 { 0u32 } else { 1u32 };

        let up = self.up_proj.forward(input, n, encoder, pipelines, pool);
        let num_elements = n * self.config.intermediate_size;
        let activated = self.apply_activation(encoder, &up, num_elements, act_type, pipelines, pool);

        
        self.down_proj.forward(&activated, n, encoder, pipelines, pool)
    }

    fn apply_activation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        num_elements: usize,
        activation_type: u32,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let key = format!("activation_{activation_type}");
        let entry = pipelines.get_or_create_default(&key, ACTIVATION_WGSL);

        let output = pool.acquire(
            (num_elements * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let params_data = [
            (num_elements as u32).to_le_bytes(),
            activation_type.to_le_bytes(),
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("activation"),
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
        pass.dispatch_workgroups(num_elements.div_ceil(256) as u32, 1, 1);

        output
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_elementwise(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        num_elements: usize,
        op: u32,
        pipelines: &mut PipelineManager,
        pool: &BufferPool,
    ) -> GpuBuf {
        let key = format!("elementwise_{op}");
        let entry = pipelines.get_or_create_default(&key, ELEMENTWISE_WGSL);

        let output = pool.acquire(
            (num_elements * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let params_data = [
            (num_elements as u32).to_le_bytes(),
            op.to_le_bytes(),
        ]
        .concat();
        let params = create_uniform_raw(&self.device, &params_data);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise"),
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
        pass.dispatch_workgroups(num_elements.div_ceil(256) as u32, 1, 1);

        output
    }

    pub fn clear_bg_cache(&mut self) {
        self.bg_cache.clear();
        self.up_proj.clear_bg_cache();
        self.down_proj.clear_bg_cache();
        if let Some(ref mut gate) = self.gate_proj {
            gate.clear_bg_cache();
        }
    }
}
