import { PipelineManager } from "../gpu/pipeline.js";
import { BufferPool } from "../gpu/buffer-pool.js";
import type { ModelConfig } from "../types.js";

// Import shader sources (bundled as strings by tsup)
import rmsnormWGSL from "../shaders/rmsnorm.wgsl";
import quantizeWGSL from "../shaders/quantize.wgsl";
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
  private normWeight: GPUBuffer; // [inDim] f32

  private inDim: number;
  private outDim: number;
  private kPacked: number;

  constructor(
    device: GPUDevice,
    pipelines: PipelineManager,
    pool: BufferPool,
    packedWeights: GPUBuffer,
    weightScales: GPUBuffer,
    normWeight: GPUBuffer,
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
    // Step 1: RMSNorm
    const normed = this.pool.acquire(
      N * this.inDim * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    this.dispatchRMSNorm(encoder, input, normed, N);

    // Step 2: Quantize (absmax int8)
    const quantized = this.pool.acquire(
      N * this.inDim * 4, // i32 stored
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const inputScales = this.pool.acquire(
      N * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.UNIFORM
    );
    this.dispatchQuantize(encoder, normed, quantized, inputScales, N);

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
    this.pool.release(normed);
    this.pool.release(quantized);
    this.pool.release(inputScales);

    return output;
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

    const paramsData = new ArrayBuffer(12);
    const paramsView = new DataView(paramsData);
    paramsView.setUint32(0, N, true);
    paramsView.setUint32(4, this.inDim, true);
    paramsView.setFloat32(8, 1e-5, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: this.normWeight } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

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

    const paramsData = new ArrayBuffer(8);
    const paramsView = new DataView(paramsData);
    paramsView.setUint32(0, N, true);
    paramsView.setUint32(4, this.inDim, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: scales } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

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

    // Params: M, K, K_packed
    const paramsData = new ArrayBuffer(12);
    const paramsView = new DataView(paramsData);
    paramsView.setUint32(0, this.outDim, true);
    paramsView.setUint32(4, this.inDim, true);
    paramsView.setUint32(8, this.kPacked, true);
    const paramsBuffer = this.createUniformBuffer(paramsData);

    // Input scale: read from first element (single token)
    const inputScaleBuffer = this.createUniformBuffer(new ArrayBuffer(4));
    // Copy from inputScales[0] to uniform
    encoder.copyBufferToBuffer(inputScales, 0, inputScaleBuffer, 0, 4);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.packedWeights } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: this.weightScales } },
        { binding: 3, resource: { buffer: paramsBuffer } },
        { binding: 4, resource: { buffer: inputScaleBuffer } },
        { binding: 5, resource: { buffer: output } },
      ],
    });

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
    const paramsBuffer = this.createUniformBuffer(paramsData);

    const inputScaleBuffer = this.createUniformBuffer(new ArrayBuffer(4));
    encoder.copyBufferToBuffer(inputScales, 0, inputScaleBuffer, 0, 4);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.packedWeights } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: this.weightScales } },
        { binding: 3, resource: { buffer: paramsBuffer } },
        { binding: 4, resource: { buffer: inputScaleBuffer } },
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
