import { PipelineManager } from "../gpu/pipeline.js";
import { BufferPool } from "../gpu/buffer-pool.js";
import { type BindGroupCache, createBGCache, clearBGCache, cachedBG } from "./bg-cache.js";
import type { ModelConfig } from "../types.js";

// Import shader sources (bundled as strings by tsup)
import rmsnormWGSL from "../shaders/rmsnorm.wgsl";
import quantizeWGSL from "../shaders/quantize.wgsl";
import rmsnormQuantizeWGSL from "../shaders/rmsnorm_quantize.wgsl";
import ternaryGemvWGSL from "../shaders/ternary_gemv.wgsl";
import ternaryGemmWGSL from "../shaders/ternary_gemm.wgsl";

/**
 * BitLinear layer: RMSNorm → Quantize → Ternary MatMul → Dequantize
 *
 * This is the core building block of BitNet. The weights are ternary {-1,0,+1}
 * packed as 2-bit values (16 per u32). Input activations are quantized to int8
 * with per-token absmax before the matmul.
 */
export class BitLinear {
  private device: GPUDevice;
  private pipelines: PipelineManager;
  private pool: BufferPool;

  // Weight buffers (on GPU)
  private packedWeights: GPUBuffer; // [outDim, inDim/16] u32
  private weightScales: GPUBuffer; // [outDim] f32
  private normWeight: GPUBuffer | null; // [inDim] f32 or null (skip RMSNorm)

  private inDim: number;
  private outDim: number;
  private kPacked: number;

  // Pre-created uniform buffers for N=1 decode (static params)
  private decodeNormUniform?: GPUBuffer;
  private decodeQuantUniform?: GPUBuffer;
  private decodeFusedUniform?: GPUBuffer;
  private decodeGemvParamsUniform?: GPUBuffer;
  private decodeGemvScaleUniform?: GPUBuffer;

  // Pre-created uniform buffers for N>1 prefill (dynamic — updated via writeBuffer)
  private prefillFusedUniform?: GPUBuffer;
  private prefillNormUniform?: GPUBuffer;
  private prefillQuantUniform?: GPUBuffer;
  private prefillGemmUniform?: GPUBuffer;

  // Bind group cache for N=1 decode
  private bgCache: BindGroupCache = createBGCache();

  constructor(
    device: GPUDevice,
    pipelines: PipelineManager,
    pool: BufferPool,
    packedWeights: GPUBuffer,
    weightScales: GPUBuffer,
    normWeight: GPUBuffer | null,
    inDim: number,
    outDim: number
  ) {
    this.device = device;
    this.pipelines = pipelines;
    this.pool = pool;
    this.packedWeights = packedWeights;
    this.weightScales = weightScales;
    this.normWeight = normWeight;
    this.inDim = inDim;
    this.outDim = outDim;
    this.kPacked = Math.ceil(inDim / 16);
  }

  /** Pre-create uniform buffers for the N=1 decode path (all static). */
  initDecodeUniforms(): void {
    if (this.normWeight) {
      // Fused RMSNorm+Quantize decode uniform
      const data = new ArrayBuffer(12);
      const v = new DataView(data);
      v.setUint32(0, 1, true);
      v.setUint32(4, this.inDim, true);
      v.setFloat32(8, 1e-5, true);
      this.decodeFusedUniform = this.createUniformBuffer(data);

      // Separate RMSNorm decode uniform (fallback)
      this.decodeNormUniform = this.createUniformBuffer(data);
    }
    {
      const data = new ArrayBuffer(8);
      const v = new DataView(data);
      v.setUint32(0, 1, true);
      v.setUint32(4, this.inDim, true);
      this.decodeQuantUniform = this.createUniformBuffer(data);
    }
    {
      const data = new ArrayBuffer(12);
      const v = new DataView(data);
      v.setUint32(0, this.outDim, true);
      v.setUint32(4, this.inDim, true);
      v.setUint32(8, this.kPacked, true);
      this.decodeGemvParamsUniform = this.createUniformBuffer(data);
    }
    this.decodeGemvScaleUniform = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Prefill uniforms (reused via writeBuffer for N>1)
    const mkBuf = (size: number) =>
      this.device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    if (this.normWeight) {
      this.prefillFusedUniform = mkBuf(12);
      this.prefillNormUniform = mkBuf(12);
    }
    this.prefillQuantUniform = mkBuf(8);
    this.prefillGemmUniform = mkBuf(16);
  }

  /**
   * Forward pass: input [N, inDim] f32 → output [N, outDim] f32
   * @param input Input buffer
   * @param N Number of tokens (sequence length)
   * @param encoder Command encoder to record dispatches
   */
  forward(
    input: GPUBuffer,
    N: number,
    encoder: GPUCommandEncoder
  ): GPUBuffer {
    // Step 1+2: RMSNorm + Quantize (fused when sub-norm exists, separate otherwise)
    let quantized: GPUBuffer;
    let inputScales: GPUBuffer;

    if (this.normWeight) {
      // Fused RMSNorm + Quantize: skip intermediate normed buffer
      quantized = this.pool.acquire(
        N * this.inDim * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      );
      inputScales = this.pool.acquire(
        N * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.UNIFORM
      );
      this.dispatchFusedNormQuantize(encoder, input, quantized, inputScales, N);
    } else {
      // No sub-norm: just quantize
      quantized = this.pool.acquire(
        N * this.inDim * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      );
      inputScales = this.pool.acquire(
        N * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.UNIFORM
      );
      this.dispatchQuantize(encoder, input, quantized, inputScales, N);
    }

    // Step 3: Ternary MatMul
    const output = this.pool.acquire(
      N * this.outDim * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    if (N === 1) {
      this.dispatchGEMV(encoder, quantized, inputScales, output);
    } else {
      this.dispatchGEMM(encoder, quantized, inputScales, output, N);
    }

    // Release intermediates
    this.pool.release(quantized);
    this.pool.release(inputScales);

    return output;
  }

  private dispatchFusedNormQuantize(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    output: GPUBuffer,
    scales: GPUBuffer,
    N: number
  ): void {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "rmsnorm_quantize",
      rmsnormQuantizeWGSL
    );

    let paramsBuffer: GPUBuffer;
    if (N === 1 && this.decodeFusedUniform) {
      paramsBuffer = this.decodeFusedUniform;
    } else if (this.prefillFusedUniform) {
      const paramsData = new ArrayBuffer(12);
      const v = new DataView(paramsData);
      v.setUint32(0, N, true);
      v.setUint32(4, this.inDim, true);
      v.setFloat32(8, 1e-5, true);
      this.device.queue.writeBuffer(this.prefillFusedUniform, 0, new Uint8Array(paramsData));
      paramsBuffer = this.prefillFusedUniform;
    } else {
      const paramsData = new ArrayBuffer(12);
      const v = new DataView(paramsData);
      v.setUint32(0, N, true);
      v.setUint32(4, this.inDim, true);
      v.setFloat32(8, 1e-5, true);
      paramsBuffer = this.createUniformBuffer(paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: this.normWeight! } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: scales } },
      { binding: 4, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1
      ? cachedBG(this.bgCache, this.device, "fused_nq", bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(N);
    pass.end();
  }

  private dispatchRMSNorm(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    output: GPUBuffer,
    N: number
  ): void {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "rmsnorm",
      rmsnormWGSL
    );

    let paramsBuffer: GPUBuffer;
    if (N === 1 && this.decodeNormUniform) {
      paramsBuffer = this.decodeNormUniform;
    } else if (this.prefillNormUniform) {
      const paramsData = new ArrayBuffer(12);
      const paramsView = new DataView(paramsData);
      paramsView.setUint32(0, N, true);
      paramsView.setUint32(4, this.inDim, true);
      paramsView.setFloat32(8, 1e-5, true);
      this.device.queue.writeBuffer(this.prefillNormUniform, 0, new Uint8Array(paramsData));
      paramsBuffer = this.prefillNormUniform;
    } else {
      const paramsData = new ArrayBuffer(12);
      const paramsView = new DataView(paramsData);
      paramsView.setUint32(0, N, true);
      paramsView.setUint32(4, this.inDim, true);
      paramsView.setFloat32(8, 1e-5, true);
      paramsBuffer = this.createUniformBuffer(paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: this.normWeight! } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1
      ? cachedBG(this.bgCache, this.device, "rmsnorm", bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(N);
    pass.end();
  }

  private dispatchQuantize(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    output: GPUBuffer,
    scales: GPUBuffer,
    N: number
  ): void {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "quantize",
      quantizeWGSL
    );

    let paramsBuffer: GPUBuffer;
    if (N === 1 && this.decodeQuantUniform) {
      paramsBuffer = this.decodeQuantUniform;
    } else if (this.prefillQuantUniform) {
      const paramsData = new ArrayBuffer(8);
      const paramsView = new DataView(paramsData);
      paramsView.setUint32(0, N, true);
      paramsView.setUint32(4, this.inDim, true);
      this.device.queue.writeBuffer(this.prefillQuantUniform, 0, new Uint8Array(paramsData));
      paramsBuffer = this.prefillQuantUniform;
    } else {
      const paramsData = new ArrayBuffer(8);
      const paramsView = new DataView(paramsData);
      paramsView.setUint32(0, N, true);
      paramsView.setUint32(4, this.inDim, true);
      paramsBuffer = this.createUniformBuffer(paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: scales } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1
      ? cachedBG(this.bgCache, this.device, "quantize", bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(N);
    pass.end();
  }

  private dispatchGEMV(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    inputScales: GPUBuffer,
    output: GPUBuffer
  ): void {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "ternary_gemv",
      ternaryGemvWGSL
    );

    let paramsBuffer: GPUBuffer;
    let inputScaleBuffer: GPUBuffer;
    if (this.decodeGemvParamsUniform && this.decodeGemvScaleUniform) {
      paramsBuffer = this.decodeGemvParamsUniform;
      inputScaleBuffer = this.decodeGemvScaleUniform;
    } else {
      const paramsData = new ArrayBuffer(12);
      const paramsView = new DataView(paramsData);
      paramsView.setUint32(0, this.outDim, true);
      paramsView.setUint32(4, this.inDim, true);
      paramsView.setUint32(8, this.kPacked, true);
      paramsBuffer = this.createUniformBuffer(paramsData);
      inputScaleBuffer = this.createUniformBuffer(new ArrayBuffer(4));
    }
    // Copy from inputScales[0] to uniform
    encoder.copyBufferToBuffer(inputScales, 0, inputScaleBuffer, 0, 4);

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.packedWeights } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: this.weightScales } },
      { binding: 3, resource: { buffer: paramsBuffer } },
      { binding: 4, resource: { buffer: inputScaleBuffer } },
      { binding: 5, resource: { buffer: output } },
    ];
    const bindGroup = cachedBG(this.bgCache, this.device, "gemv", bindGroupLayout, entries);

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.outDim);
    pass.end();
  }

  private dispatchGEMM(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    inputScales: GPUBuffer,
    output: GPUBuffer,
    N: number
  ): void {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "ternary_gemm",
      ternaryGemmWGSL
    );

    // Params: M, N, K, K_packed
    const paramsData = new ArrayBuffer(16);
    const paramsView = new DataView(paramsData);
    paramsView.setUint32(0, this.outDim, true);
    paramsView.setUint32(4, N, true);
    paramsView.setUint32(8, this.inDim, true);
    paramsView.setUint32(12, this.kPacked, true);

    let paramsBuffer: GPUBuffer;
    if (this.prefillGemmUniform) {
      this.device.queue.writeBuffer(this.prefillGemmUniform, 0, new Uint8Array(paramsData));
      paramsBuffer = this.prefillGemmUniform;
    } else {
      paramsBuffer = this.createUniformBuffer(paramsData);
    }

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.packedWeights } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: this.weightScales } },
        { binding: 3, resource: { buffer: paramsBuffer } },
        { binding: 4, resource: { buffer: inputScales } },
        { binding: 5, resource: { buffer: output } },
      ],
    });

    const wgM = Math.ceil(this.outDim / 64);
    const wgN = Math.ceil(N / 64);

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgM, wgN);
    pass.end();
  }

  /** Clear the bind group cache (call on KV cache reset). */
  clearBGCache(): void {
    clearBGCache(this.bgCache);
  }

  private createUniformBuffer(data: ArrayBuffer): GPUBuffer {
    const size = Math.max(Math.ceil(data.byteLength / 4) * 4, 4);
    const buffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
    buffer.unmap();
    return buffer;
  }
}
