use std::collections::HashMap;

/// Bind Group Cache — caches wgpu::BindGroup objects by string key.
///
/// For MVP we simply cache bind groups by their string ID.
/// A bind group is invalidated (and recreated) whenever clear() is called
/// (e.g., on KV cache reset). This avoids the need for buffer identity tracking.
pub struct BgCache {
    entries: HashMap<String, wgpu::BindGroup>,
}

impl Default for BgCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BgCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}
