import { PipelineManager } from "../gpu/pipeline.js";
import { BufferPool, createUniformBuffer } from "../gpu/buffer-pool.js";
import { Attention } from "./attention.js";
import { FFN } from "./ffn.js";
import { type BindGroupCache, createBGCache, clearBGCache, cachedBG } from "./bg-cache.js";
import type { ModelConfig, KVCache } from "../types.js";

import rmsnormWGSL from "../shaders/rmsnorm.wgsl";
import elementwiseWGSL from "../shaders/elementwise.wgsl";

/**
 * Single transformer block:
 *   residual = x
 *   x = attention(input_layernorm(x)) + residual
 *   residual = x
 *   x = ffn(post_attention_layernorm(x)) + residual
 */
export class TransformerBlock {
  private device: GPUDevice;
  private pipelines: PipelineManager;
  private pool: BufferPool;
  private config: ModelConfig;

  private inputLayerNorm: GPUBuffer; // [hidden] f32
  private postAttnLayerNorm: GPUBuffer; // [hidden] f32
  private attention: Attention;
  private ffn: FFN;

  // Pre-created uniform buffers for N=1 decode (static params, shared within block)
  private decodeNormUniform?: GPUBuffer;
  private decodeAddUniform?: GPUBuffer;

  // Pre-created uniform buffers for N>1 prefill (dynamic — updated via writeBuffer)
  private prefillNormUniform?: GPUBuffer;
  private prefillAddUniform?: GPUBuffer;

  // Bind group cache for N=1 decode
  private bgCache: BindGroupCache = createBGCache();

  constructor(
    device: GPUDevice,
    pipelines: PipelineManager,
    pool: BufferPool,
    config: ModelConfig,
    inputLayerNorm: GPUBuffer,
    postAttnLayerNorm: GPUBuffer,
    attention: Attention,
    ffn: FFN
  ) {
    this.device = device;
    this.pipelines = pipelines;
    this.pool = pool;
    this.config = config;
    this.inputLayerNorm = inputLayerNorm;
    this.postAttnLayerNorm = postAttnLayerNorm;
    this.attention = attention;
    this.ffn = ffn;
  }

  /** Pre-create uniform buffers for N=1 decode path (all static). */
  initDecodeUniforms(maxSeqLen: number): void {
    {
      const data = new ArrayBuffer(12);
      const v = new DataView(data);
      v.setUint32(0, 1, true);
      v.setUint32(4, this.config.hiddenSize, true);
      v.setFloat32(8, this.config.rmsNormEps, true);
      this.decodeNormUniform = createUniformBuffer(this.device, data);
    }
    {
      const data = new ArrayBuffer(8);
      const v = new DataView(data);
      v.setUint32(0, this.config.hiddenSize, true);
      v.setUint32(4, 0, true); // add
      this.decodeAddUniform = createUniformBuffer(this.device, data);
    }

    // Prefill uniforms (reused via writeBuffer for N>1)
    const mkBuf = (size: number) =>
      this.device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.prefillNormUniform = mkBuf(12);
    this.prefillAddUniform = mkBuf(8);

    this.attention.initDecodeUniforms(maxSeqLen);
    this.ffn.initDecodeUniforms();
  }

  /**
   * Forward pass through one transformer block.
   * @param input [N, hidden] f32
   * @param N sequence length
   * @param kvCache KV cache for this layer
   * @param encoder GPU command encoder
   * @returns output [N, hidden] f32
   */
  forward(
    input: GPUBuffer,
    N: number,
    kvCache: KVCache,
    encoder: GPUCommandEncoder
  ): GPUBuffer {
    const hidden = this.config.hiddenSize;

    // Self-attention with residual
    const normedAttn = this.dispatchRMSNorm(
      encoder,
      input,
      this.inputLayerNorm,
      N,
      "attnNorm"
    );
    const attnOut = this.attention.forward(normedAttn, N, kvCache, encoder);
    this.pool.release(normedAttn);

    const residual1 = this.dispatchAdd(
      encoder,
      input,
      attnOut,
      N * hidden,
      N,
      "attnAdd"
    );
    this.pool.release(attnOut);

    // FFN with residual
    const normedFFN = this.dispatchRMSNorm(
      encoder,
      residual1,
      this.postAttnLayerNorm,
      N,
      "ffnNorm"
    );
    const ffnOut = this.ffn.forward(normedFFN, N, encoder);
    this.pool.release(normedFFN);

    const output = this.dispatchAdd(
      encoder,
      residual1,
      ffnOut,
      N * hidden,
      N,
      "ffnAdd"
    );
    this.pool.release(residual1);
    this.pool.release(ffnOut);

    return output;
  }

  private dispatchRMSNorm(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    weight: GPUBuffer,
    N: number,
    bgId?: string
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "rmsnorm",
      rmsnormWGSL
    );

    const hidden = this.config.hiddenSize;
    const output = this.pool.acquire(
      N * hidden * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    let paramsBuffer: GPUBuffer;
    if (N === 1 && this.decodeNormUniform) {
      paramsBuffer = this.decodeNormUniform;
    } else if (this.prefillNormUniform) {
      const paramsData = new ArrayBuffer(12);
      const v = new DataView(paramsData);
      v.setUint32(0, N, true);
      v.setUint32(4, hidden, true);
      v.setFloat32(8, this.config.rmsNormEps, true);
      this.device.queue.writeBuffer(this.prefillNormUniform, 0, new Uint8Array(paramsData));
      paramsBuffer = this.prefillNormUniform;
    } else {
      const paramsData = new ArrayBuffer(12);
      const v = new DataView(paramsData);
      v.setUint32(0, N, true);
      v.setUint32(4, hidden, true);
      v.setFloat32(8, this.config.rmsNormEps, true);
      paramsBuffer = createUniformBuffer(this.device, paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: weight } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1 && bgId
      ? cachedBG(this.bgCache, this.device, bgId, bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(N);
    pass.end();

    return output;
  }

  private dispatchAdd(
    encoder: GPUCommandEncoder,
    a: GPUBuffer,
    b: GPUBuffer,
    numElements: number,
    N: number,
    bgId?: string
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "elementwise_0",
      elementwiseWGSL
    );

    const output = this.pool.acquire(
      numElements * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    let paramsBuffer: GPUBuffer;
    if (N === 1 && this.decodeAddUniform) {
      paramsBuffer = this.decodeAddUniform;
    } else if (this.prefillAddUniform) {
      const paramsData = new ArrayBuffer(8);
      const v = new DataView(paramsData);
      v.setUint32(0, numElements, true);
      v.setUint32(4, 0, true); // add
      this.device.queue.writeBuffer(this.prefillAddUniform, 0, new Uint8Array(paramsData));
      paramsBuffer = this.prefillAddUniform;
    } else {
      const paramsData = new ArrayBuffer(8);
      const v = new DataView(paramsData);
      v.setUint32(0, numElements, true);
      v.setUint32(4, 0, true); // add
      paramsBuffer = createUniformBuffer(this.device, paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: a } },
      { binding: 1, resource: { buffer: b } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1 && bgId
      ? cachedBG(this.bgCache, this.device, bgId, bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(numElements / 256));
    pass.end();

    return output;
  }

  /** Clear the bind group cache (call on KV cache reset). */
  clearBGCache(): void {
    clearBGCache(this.bgCache);
    this.attention.clearBGCache();
    this.ffn.clearBGCache();
  }

  /** Destroy pre-allocated attention buffers. */
  destroyPreAllocated(): void {
    this.attention.destroyPreAllocated();
  }

}
