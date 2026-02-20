use std::collections::HashMap;
use std::sync::Arc;

use wgpu::BufferUsages;

use crate::gpu::buffer_pool::GpuBuf;

/// Weight buffer management: maps tensor names to GPU buffers.
pub struct WeightStore {
    buffers: HashMap<String, GpuBuf>,
    device: Arc<wgpu::Device>,
}

impl WeightStore {
    pub fn new(device: Arc<wgpu::Device>, _queue: Arc<wgpu::Queue>) -> Self {
        Self {
            buffers: HashMap::new(),
            device,
        }
    }

    /// Upload a tensor to the GPU as a storage buffer.
    pub fn upload(&mut self, name: &str, data: &[u8]) -> GpuBuf {
        let size = data.len().max(4) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = buffer.slice(..).get_mapped_range_mut();
            view[..data.len()].copy_from_slice(data);
        }
        buffer.unmap();
        let buf = Arc::new(buffer);
        self.buffers.insert(name.to_string(), Arc::clone(&buf));
        buf
    }

    /// Upload a large tensor, sharding if it exceeds the binding limit.
    pub fn upload_sharded(
        &mut self,
        name: &str,
        data: &[u8],
        max_binding_size: u32,
    ) -> Vec<GpuBuf> {
        let max = max_binding_size as usize;
        if data.len() <= max {
            return vec![self.upload(name, data)];
        }

        let mut shards = Vec::new();
        let mut offset = 0usize;
        let mut shard_idx = 0;
        while offset < data.len() {
            let end = (offset + max).min(data.len());
            let shard_name = format!("{name}.shard_{shard_idx}");
            shards.push(self.upload(&shard_name, &data[offset..end]));
            offset = end;
            shard_idx += 1;
        }
        // Also store first shard under original name
        if !shards.is_empty() && !self.buffers.contains_key(name) {
            self.buffers
                .insert(name.to_string(), Arc::clone(&shards[0]));
        }
        shards
    }

    pub fn get(&self, name: &str) -> Option<&GpuBuf> {
        self.buffers.get(name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.buffers.contains_key(name)
    }
}
