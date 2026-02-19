import { PipelineManager } from "../gpu/pipeline.js";
import { BufferPool } from "../gpu/buffer-pool.js";
import { Attention } from "./attention.js";
import { FFN } from "./ffn.js";
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
      N
    );
    const attnOut = this.attention.forward(normedAttn, N, kvCache, encoder);
    this.pool.release(normedAttn);

    const residual1 = this.dispatchAdd(
      encoder,
      input,
      attnOut,
      N * hidden
    );
    this.pool.release(attnOut);

    // FFN with residual
    const normedFFN = this.dispatchRMSNorm(
      encoder,
      residual1,
      this.postAttnLayerNorm,
      N
    );
    const ffnOut = this.ffn.forward(normedFFN, N, encoder);
    this.pool.release(normedFFN);

    const output = this.dispatchAdd(
      encoder,
      residual1,
      ffnOut,
      N * hidden
    );
    this.pool.release(residual1);
    this.pool.release(ffnOut);

    return output;
  }

  private dispatchRMSNorm(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    weight: GPUBuffer,
    N: number
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

    const paramsData = new ArrayBuffer(12);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, hidden, true);
    v.setFloat32(8, this.config.rmsNormEps, true);
    const paramsBuffer = this.createUniform(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

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
    numElements: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "elementwise_0",
      elementwiseWGSL
    );

    const output = this.pool.acquire(
      numElements * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(8);
    const v = new DataView(paramsData);
    v.setUint32(0, numElements, true);
    v.setUint32(4, 0, true); // add
    const paramsBuffer = this.createUniform(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(numElements / 256));
    pass.end();

    return output;
  }

  private createUniform(data: ArrayBuffer): GPUBuffer {
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
