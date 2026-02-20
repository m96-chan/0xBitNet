/**
 * Bind Group Cache — caches GPUBindGroup objects by buffer identity.
 *
 * During N=1 decode, the buffer pool returns the same physical GPUBuffer
 * objects deterministically (identical acquire/release pattern). This means
 * bind group entries reference the same buffers every token, so we can
 * cache and reuse the bind group instead of calling createBindGroup (~5-10μs
 * IPC overhead each).
 *
 * Cache key: string ID (e.g. "rmsnorm", "gemv")
 * Cache hit: all GPUBuffer refs in entries match the cached version
 * Cache miss: create new bind group, store it + buffer refs
 */

export type BindGroupCache = Map<string, { bg: GPUBindGroup; bufs: GPUBuffer[] }>;

/** Create a new empty bind group cache. */
export function createBGCache(): BindGroupCache {
  return new Map();
}

/** Clear all entries from a bind group cache. */
export function clearBGCache(cache: BindGroupCache): void {
  cache.clear();
}

/**
 * Get or create a cached bind group.
 *
 * Extracts GPUBuffer references from `entries`, compares with the cached
 * entry for `id`. Returns the cached bind group on match, otherwise
 * creates a new one and caches it.
 */
export function cachedBG(
  cache: BindGroupCache,
  device: GPUDevice,
  id: string,
  layout: GPUBindGroupLayout,
  entries: GPUBindGroupEntry[]
): GPUBindGroup {
  const bufs = entries.map((e) => (e.resource as GPUBufferBinding).buffer);
  const cached = cache.get(id);

  if (cached && cached.bufs.length === bufs.length) {
    let match = true;
    for (let i = 0; i < bufs.length; i++) {
      if (cached.bufs[i] !== bufs[i]) {
        match = false;
        break;
      }
    }
    if (match) return cached.bg;
  }

  const bg = device.createBindGroup({ layout, entries });
  cache.set(id, { bg, bufs });
  return bg;
}
