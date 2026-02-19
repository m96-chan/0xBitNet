export interface PipelineEntry {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Manages creation and caching of WebGPU compute pipelines.
 * Each unique (shader source + entry point) pair produces one cached pipeline.
 */
export class PipelineManager {
  private cache = new Map<string, PipelineEntry>();
  private device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Get or create a compute pipeline from WGSL source code.
   * @param key Unique cache key for this pipeline
   * @param wgsl WGSL shader source
   * @param entryPoint Compute entry point name (default: "main")
   * @param constants Pipeline-overridable constants
   */
  getOrCreate(
    key: string,
    wgsl: string,
    entryPoint = "main",
    constants?: Record<string, number>
  ): PipelineEntry {
    const cacheKey = constants
      ? `${key}:${JSON.stringify(constants)}`
      : key;

    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    const shaderModule = this.device.createShaderModule({ code: wgsl });

    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint,
        constants,
      },
    });

    const bindGroupLayout = pipeline.getBindGroupLayout(0);
    const entry: PipelineEntry = { pipeline, bindGroupLayout };
    this.cache.set(cacheKey, entry);
    return entry;
  }

  /** Clear all cached pipelines */
  clear(): void {
    this.cache.clear();
  }
}
