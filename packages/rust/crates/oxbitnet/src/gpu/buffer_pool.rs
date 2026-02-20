use std::sync::Arc;

use wgpu::BufferUsages;

/// A shareable reference to a GPU buffer.
pub type GpuBuf = Arc<wgpu::Buffer>;

/// GPU buffer pool / factory.
///
/// For MVP, this simply creates fresh buffers. A future optimization
/// can add size-bucketed reuse (like the TS BufferPool).
pub struct BufferPool {
    device: Arc<wgpu::Device>,
    alignment: u64,
}

impl BufferPool {
    pub fn new(device: Arc<wgpu::Device>, alignment: u64) -> Self {
        Self { device, alignment }
    }

    fn align_size(&self, size: u64) -> u64 {
        ((size + self.alignment - 1) / self.alignment) * self.alignment
    }

    /// Create a buffer of at least `size` bytes with the given usage flags.
    pub fn acquire(&self, size: u64, usage: BufferUsages) -> GpuBuf {
        let aligned = self.align_size(size.max(4));
        Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: aligned,
            usage,
            mapped_at_creation: false,
        }))
    }

    /// Release a buffer (currently a no-op; buffer is destroyed when Arc drops).
    pub fn release(&self, _buffer: GpuBuf) {
        // Drop — destroyed when last Arc ref is gone.
    }
}
