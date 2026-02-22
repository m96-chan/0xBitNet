import type { BufferEntry } from "../types.js";

/** Create a GPU uniform buffer from an ArrayBuffer, with mappedAtCreation. */
export function createUniformBuffer(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
  const size = Math.max(Math.ceil(data.byteLength / 4) * 4, 4);
  const buffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
  buffer.unmap();
  return buffer;
}

/** Next power of 2 >= n (returns n if already a power of 2). */
function nextPow2(n: number): number {
  if (n <= 1) return 1;
  return 1 << (32 - Math.clz32(n - 1));
}

/** Composite key for a bucket: usage flags + size class. */
function bucketKey(usage: number, sizeClass: number): string {
  return `${usage}:${sizeClass}`;
}

export interface BufferPoolStats {
  totalBuffers: number;
  inUse: number;
  totalBytes: number;
}

/**
 * GPU buffer pool for reusing temporary compute buffers.
 * Buckets buffers by (usage, sizeClass) where sizeClass is the next power of 2
 * of the aligned size. Within each bucket, buffers are exact-size matches
 * so acquire is O(1) pop from the free list.
 */
export class BufferPool {
  private device: GPUDevice;
  private alignment: number;
  // Bucket key → list of free (reusable) buffers
  private free = new Map<string, BufferEntry[]>();
  // All tracked buffers (for release lookup and stats)
  private bufferToEntry = new Map<GPUBuffer, BufferEntry & { key: string }>();

  constructor(device: GPUDevice, alignment = 256) {
    this.device = device;
    this.alignment = alignment;
  }

  /** Round size up to the nearest alignment boundary */
  private alignSize(size: number): number {
    return Math.ceil(size / this.alignment) * this.alignment;
  }

  /**
   * Acquire a buffer of at least `size` bytes with the given usage flags.
   * Returns a pooled buffer if one is available, otherwise creates a new one.
   */
  acquire(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const aligned = this.alignSize(size);
    const sizeClass = nextPow2(aligned);
    const key = bucketKey(usage, sizeClass);

    const bucket = this.free.get(key);
    if (bucket && bucket.length > 0) {
      const entry = bucket.pop()!;
      entry.inUse = true;
      return entry.buffer;
    }

    const buffer = this.device.createBuffer({ size: sizeClass, usage });
    const entry = { buffer, size: sizeClass, inUse: true, key };
    this.bufferToEntry.set(buffer, entry);
    return buffer;
  }

  /** Release a buffer back to the pool for reuse. */
  release(buffer: GPUBuffer): void {
    const entry = this.bufferToEntry.get(buffer);
    if (!entry || !entry.inUse) return;
    entry.inUse = false;

    let bucket = this.free.get(entry.key);
    if (!bucket) {
      bucket = [];
      this.free.set(entry.key, bucket);
    }
    bucket.push(entry);
  }

  /** Return pool statistics. */
  stats(): BufferPoolStats {
    let totalBuffers = 0;
    let inUse = 0;
    let totalBytes = 0;
    for (const entry of this.bufferToEntry.values()) {
      totalBuffers++;
      totalBytes += entry.size;
      if (entry.inUse) inUse++;
    }
    return { totalBuffers, inUse, totalBytes };
  }

  /** Destroy all free (not in-use) buffers to reclaim GPU memory. */
  trim(): void {
    for (const [key, bucket] of this.free) {
      for (const entry of bucket) {
        entry.buffer.destroy();
        this.bufferToEntry.delete(entry.buffer);
      }
      bucket.length = 0;
    }
    this.free.clear();
  }

  /** Destroy all pooled buffers and clear the pool. */
  destroy(): void {
    for (const entry of this.bufferToEntry.values()) {
      entry.buffer.destroy();
    }
    this.bufferToEntry.clear();
    this.free.clear();
  }
}
