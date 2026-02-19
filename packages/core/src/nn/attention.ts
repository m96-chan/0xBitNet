import { PipelineManager } from "../gpu/pipeline.js";
import { BufferPool } from "../gpu/buffer-pool.js";
import { BitLinear } from "./bitlinear.js";
import type { ModelConfig, KVCache } from "../types.js";
import { headDim } from "../model/config.js";

import ropeWGSL from "../shaders/rope.wgsl";
import softmaxWGSL from "../shaders/softmax.wgsl";
import attentionWGSL from "../shaders/attention.wgsl";

/**
 * Multi-Head Attention with support for GQA (Grouped Query Attention).
 *
 * Components:
 * - Q/K/V projections via BitLinear
 * - RoPE (Rotary Position Embeddings)
 * - Scaled dot-product attention with causal mask
 * - Output projection via BitLinear
 * - KV-cache for autoregressive generation
 */
export class Attention {
  private device: GPUDevice;
  private pipelines: PipelineManager;
  private pool: BufferPool;
  private config: ModelConfig;
  private hDim: number;

  private qProj: BitLinear;
  private kProj: BitLinear;
  private vProj: BitLinear;
  private oProj: BitLinear;

  constructor(
    device: GPUDevice,
    pipelines: PipelineManager,
    pool: BufferPool,
    config: ModelConfig,
    qProj: BitLinear,
    kProj: BitLinear,
    vProj: BitLinear,
    oProj: BitLinear
  ) {
    this.device = device;
    this.pipelines = pipelines;
    this.pool = pool;
    this.config = config;
    this.hDim = headDim(config);
    this.qProj = qProj;
    this.kProj = kProj;
    this.vProj = vProj;
    this.oProj = oProj;
  }

  /**
   * Forward pass.
   * @param input [N, hidden] f32
   * @param N sequence length
   * @param kvCache KV cache for autoregressive decoding
   * @param encoder GPU command encoder
   * @returns output [N, hidden] f32
   */
  forward(
    input: GPUBuffer,
    N: number,
    kvCache: KVCache,
    encoder: GPUCommandEncoder
  ): GPUBuffer {
    const { numAttentionHeads, numKeyValueHeads, hiddenSize } = this.config;

    // Q/K/V projections via BitLinear
    const qBuf = this.qProj.forward(input, N, encoder);
    const kBuf = this.kProj.forward(input, N, encoder);
    const vBuf = this.vProj.forward(input, N, encoder);

    // Apply RoPE to Q and K
    const qRoped = this.applyRoPE(
      encoder,
      qBuf,
      N,
      numAttentionHeads,
      kvCache.seqLen
    );
    const kRoped = this.applyRoPE(
      encoder,
      kBuf,
      N,
      numKeyValueHeads,
      kvCache.seqLen
    );

    this.pool.release(qBuf);
    this.pool.release(kBuf);

    // Update KV cache
    this.appendToCache(encoder, kRoped, vBuf, kvCache, N);
    this.pool.release(kRoped);
    this.pool.release(vBuf);

    const totalSeq = kvCache.seqLen + N;

    // Compute attention scores: Q @ K^T * scale
    const scores = this.computeScores(
      encoder,
      qRoped,
      kvCache.key,
      N,
      totalSeq
    );
    this.pool.release(qRoped);

    // Softmax over scores
    const attnWeights = this.applySoftmax(
      encoder,
      scores,
      numAttentionHeads * N,
      totalSeq
    );
    this.pool.release(scores);

    // Attention output: weights @ V
    const attnOutput = this.computeAttnV(
      encoder,
      attnWeights,
      kvCache.value,
      N,
      totalSeq
    );
    this.pool.release(attnWeights);

    // Output projection via BitLinear
    const output = this.oProj.forward(attnOutput, N, encoder);
    this.pool.release(attnOutput);

    return output;
  }

  private applyRoPE(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    N: number,
    numHeads: number,
    posOffset: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "rope",
      ropeWGSL
    );

    const outputSize = N * numHeads * this.hDim * 4;
    const output = this.pool.acquire(
      outputSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(20);
    const paramsView = new DataView(paramsData);
    paramsView.setUint32(0, N, true);
    paramsView.setUint32(4, numHeads, true);
    paramsView.setUint32(8, this.hDim, true);
    paramsView.setUint32(12, posOffset, true);
    paramsView.setFloat32(16, this.config.ropeTheta, true);
    const paramsBuffer = this.createUniform(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const totalPairs = N * numHeads * (this.hDim / 2);
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalPairs / 256));
    pass.end();

    return output;
  }

  private appendToCache(
    encoder: GPUCommandEncoder,
    k: GPUBuffer,
    v: GPUBuffer,
    cache: KVCache,
    N: number
  ): void {
    const kvSize =
      N * this.config.numKeyValueHeads * this.hDim * 4;
    const offset = cache.seqLen * this.config.numKeyValueHeads * this.hDim * 4;

    encoder.copyBufferToBuffer(k, 0, cache.key, offset, kvSize);
    encoder.copyBufferToBuffer(v, 0, cache.value, offset, kvSize);
  }

  private computeScores(
    encoder: GPUCommandEncoder,
    Q: GPUBuffer,
    K: GPUBuffer,
    N: number,
    S: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "attention_scores",
      attentionWGSL,
      "compute_scores"
    );

    const { numAttentionHeads, numKeyValueHeads } = this.config;
    const scoresSize = numAttentionHeads * N * S * 4;
    const scores = this.pool.acquire(
      scoresSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(24);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, S, true);
    v.setUint32(8, numAttentionHeads, true);
    v.setUint32(12, numKeyValueHeads, true);
    v.setUint32(16, this.hDim, true);
    v.setFloat32(20, 1.0 / Math.sqrt(this.hDim), true);
    const paramsBuffer = this.createUniform(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: Q } },
        { binding: 1, resource: { buffer: K } },
        { binding: 2, resource: { buffer: scores } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(N / 16),
      Math.ceil(S / 16),
      numAttentionHeads
    );
    pass.end();

    return scores;
  }

  private applySoftmax(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    N: number,
    D: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "softmax",
      softmaxWGSL
    );

    const output = this.pool.acquire(
      N * D * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(8);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, D, true);
    const paramsBuffer = this.createUniform(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(N);
    pass.end();

    return output;
  }

  private computeAttnV(
    encoder: GPUCommandEncoder,
    attn: GPUBuffer,
    V: GPUBuffer,
    N: number,
    S: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "attn_v",
      attentionWGSL,
      "attn_v"
    );

    const { numAttentionHeads, numKeyValueHeads } = this.config;
    const outputSize = N * numAttentionHeads * this.hDim * 4;
    const output = this.pool.acquire(
      outputSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(20);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, S, true);
    v.setUint32(8, numAttentionHeads, true);
    v.setUint32(12, numKeyValueHeads, true);
    v.setUint32(16, this.hDim, true);
    const paramsBuffer = this.createUniform(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: attn } },
        { binding: 1, resource: { buffer: V } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const total = N * numAttentionHeads * this.hDim;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 256));
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

/** Allocate a fresh KV cache for a given config and max sequence length */
export function createKVCache(
  device: GPUDevice,
  config: ModelConfig,
  maxSeqLen: number
): KVCache {
  const kvSize =
    maxSeqLen * config.numKeyValueHeads * headDim(config) * 4;
  const key = device.createBuffer({
    size: kvSize,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const value = device.createBuffer({
    size: kvSize,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  return { key, value, seqLen: 0, maxSeqLen };
}
