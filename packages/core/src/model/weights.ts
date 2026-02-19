/**
 * Weight buffer management: maps tensor names to GPU buffers.
 * Handles weight sharding for tensors exceeding maxStorageBufferBindingSize.
 */
export class WeightStore {
  private buffers = new Map<string, GPUBuffer>();
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /** Upload a tensor to the GPU as a storage buffer */
  upload(name: string, data: ArrayBuffer): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: Math.max(data.byteLength, 4), // WebGPU minimum buffer size
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
    buffer.unmap();
    this.buffers.set(name, buffer);
    return buffer;
  }

  /**
   * Upload a large tensor in shards if it exceeds the binding limit.
   * Returns an array of shard buffers.
   */
  uploadSharded(
    name: string,
    data: ArrayBuffer,
    maxBindingSize: number
  ): GPUBuffer[] {
    if (data.byteLength <= maxBindingSize) {
      return [this.upload(name, data)];
    }

    const shards: GPUBuffer[] = [];
    let offset = 0;
    let shardIdx = 0;
    while (offset < data.byteLength) {
      const end = Math.min(offset + maxBindingSize, data.byteLength);
      const shard = data.slice(offset, end);
      const shardName = `${name}.shard_${shardIdx}`;
      shards.push(this.upload(shardName, shard));
      offset = end;
      shardIdx++;
    }
    // Also store first shard under the original name for non-sharding-aware code
    if (shards.length > 0 && !this.buffers.has(name)) {
      this.buffers.set(name, shards[0]);
    }
    return shards;
  }

  /** Get a buffer by tensor name */
  get(name: string): GPUBuffer | undefined {
    return this.buffers.get(name);
  }

  /** Check if a tensor is loaded */
  has(name: string): boolean {
    return this.buffers.has(name);
  }

  /** Destroy all GPU buffers */
  destroy(): void {
    for (const buffer of this.buffers.values()) {
      buffer.destroy();
    }
    this.buffers.clear();
  }
}
