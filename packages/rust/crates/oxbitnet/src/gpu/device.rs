use std::sync::Arc;

use crate::error::{BitNetError, Result};

pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

/// Initialize wgpu device and queue with maximum limits.
pub async fn init_gpu() -> Result<GpuContext> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| BitNetError::Gpu(format!("Failed to get adapter: {e}")))?;

    let mut required_limits = wgpu::Limits::default();
    let adapter_limits = adapter.limits();

    required_limits.max_buffer_size = adapter_limits.max_buffer_size;
    required_limits.max_storage_buffer_binding_size =
        adapter_limits.max_storage_buffer_binding_size;
    required_limits.max_storage_buffers_per_shader_stage =
        adapter_limits.max_storage_buffers_per_shader_stage;
    required_limits.max_compute_workgroup_size_x =
        adapter_limits.max_compute_workgroup_size_x;
    required_limits.max_compute_workgroup_size_y =
        adapter_limits.max_compute_workgroup_size_y;
    required_limits.max_compute_workgroup_size_z =
        adapter_limits.max_compute_workgroup_size_z;
    required_limits.max_compute_invocations_per_workgroup =
        adapter_limits.max_compute_invocations_per_workgroup;
    required_limits.max_compute_workgroup_storage_size =
        adapter_limits.max_compute_workgroup_storage_size;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("oxbitnet"),
            required_features: wgpu::Features::empty(),
            required_limits,
            ..Default::default()
        })
        .await?;

    Ok(GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
    })
}
