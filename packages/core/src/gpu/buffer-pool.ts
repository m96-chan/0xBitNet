import type { BufferEntry } from "../types.js";

/**
 * GPU buffer pool for reusing temporary compute buffers.
 * Reduces allocation churn during inference by recycling buffers
 * that match the requested size (rounded up to alignment).
 */
export class BufferPool {
  private device: GPUDevice;
  private pools = new Map<number, BufferEntry[]>();
  private bufferToEntry = new Map<GPUBuffer, BufferEntry>();
  private alignment: number;

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
    const pool = this.pools.get(usage);

    if (pool) {
      for (const entry of pool) {
        if (!entry.inUse && entry.size >= aligned) {
          entry.inUse = true;
          return entry.buffer;
        }
      }
    }

    const buffer = this.device.createBuffer({ size: aligned, usage });
    const entry: BufferEntry = { buffer, size: aligned, inUse: true };
    this.bufferToEntry.set(buffer, entry);

    if (!pool) {
      this.pools.set(usage, [entry]);
    } else {
      pool.push(entry);
    }

    return buffer;
  }

  /** Release a buffer back to the pool for reuse. */
  release(buffer: GPUBuffer): void {
    const entry = this.bufferToEntry.get(buffer);
    if (entry) {
      entry.inUse = false;
    }
  }

  /** Destroy all pooled buffers and clear the pool. */
  destroy(): void {
    for (const pool of this.pools.values()) {
      for (const entry of pool) {
        entry.buffer.destroy();
      }
    }
    this.pools.clear();
    this.bufferToEntry.clear();
  }
}
