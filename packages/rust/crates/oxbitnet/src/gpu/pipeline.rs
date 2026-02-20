use std::collections::HashMap;
use std::sync::Arc;

pub struct PipelineEntry {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

/// Manages creation and caching of wgpu compute pipelines.
pub struct PipelineManager {
    cache: HashMap<String, Arc<PipelineEntry>>,
    device: Arc<wgpu::Device>,
}

impl PipelineManager {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            cache: HashMap::new(),
            device,
        }
    }

    /// Get or create a compute pipeline from WGSL source code.
    pub fn get_or_create(
        &mut self,
        key: &str,
        wgsl: &str,
        entry_point: &str,
        _constants: Option<&HashMap<String, f64>>,
    ) -> Arc<PipelineEntry> {
        let cache_key = key.to_string();

        if let Some(entry) = self.cache.get(&cache_key) {
            return Arc::clone(entry);
        }

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(key),
                layout: None,
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let entry = Arc::new(PipelineEntry {
            pipeline,
            bind_group_layout,
        });
        self.cache.insert(cache_key, Arc::clone(&entry));
        entry
    }

    /// Convenience: get_or_create with "main" entry point and no constants.
    pub fn get_or_create_default(&mut self, key: &str, wgsl: &str) -> Arc<PipelineEntry> {
        self.get_or_create(key, wgsl, "main", None)
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
