import { PipelineManager } from "../gpu/pipeline.js";
import { BufferPool } from "../gpu/buffer-pool.js";
import { BitLinear } from "./bitlinear.js";
import { type BindGroupCache, createBGCache, clearBGCache, cachedBG } from "./bg-cache.js";
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

  // Pre-allocated uniform buffers for N=1 decode (dynamic — updated via writeBuffer)
  private decodeRopeQUniform?: GPUBuffer;
  private decodeRopeKUniform?: GPUBuffer;
  private decodeScoresUniform?: GPUBuffer;
  private decodeSoftmaxUniform?: GPUBuffer;
  private decodeAttnVUniform?: GPUBuffer;

  // Pre-allocated uniform buffers for N>1 prefill (dynamic — updated via writeBuffer)
  private prefillRopeQUniform?: GPUBuffer;
  private prefillRopeKUniform?: GPUBuffer;
  private prefillScoresUniform?: GPUBuffer;
  private prefillSoftmaxUniform?: GPUBuffer;
  private prefillAttnVUniform?: GPUBuffer;

  // Pre-allocated score/attnWeight buffers for N=1 decode (sized to maxSeqLen)
  private decodeScoresBuf?: GPUBuffer;
  private decodeAttnWeightsBuf?: GPUBuffer;

  // Bind group cache for N=1 decode
  private bgCache: BindGroupCache = createBGCache();

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

  /** Pre-allocate uniform buffers for N=1 decode (updated via writeBuffer each token). */
  initDecodeUniforms(maxSeqLen: number): void {
    const mkBuf = (size: number) =>
      this.device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.decodeRopeQUniform = mkBuf(20);
    this.decodeRopeKUniform = mkBuf(20);
    this.decodeScoresUniform = mkBuf(24);
    this.decodeSoftmaxUniform = mkBuf(8);
    this.decodeAttnVUniform = mkBuf(20);

    // Prefill uniforms (reused via writeBuffer for N>1)
    this.prefillRopeQUniform = mkBuf(20);
    this.prefillRopeKUniform = mkBuf(20);
    this.prefillScoresUniform = mkBuf(24);
    this.prefillSoftmaxUniform = mkBuf(8);
    this.prefillAttnVUniform = mkBuf(20);

    // Pre-allocate score/attnWeight buffers at maxSeqLen so buffer identity
    // is stable across decode tokens (totalSeq grows but buffer stays the same)
    const maxScoresSize = this.config.numAttentionHeads * maxSeqLen * 4;
    this.decodeScoresBuf = this.device.createBuffer({
      size: maxScoresSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.decodeAttnWeightsBuf = this.device.createBuffer({
      size: maxScoresSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.qProj.initDecodeUniforms();
    this.kProj.initDecodeUniforms();
    this.vProj.initDecodeUniforms();
    this.oProj.initDecodeUniforms();
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
      kvCache.seqLen,
      N === 1 ? this.decodeRopeQUniform : this.prefillRopeQUniform,
      "ropeQ"
    );
    const kRoped = this.applyRoPE(
      encoder,
      kBuf,
      N,
      numKeyValueHeads,
      kvCache.seqLen,
      N === 1 ? this.decodeRopeKUniform : this.prefillRopeKUniform,
      "ropeK"
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
      totalSeq,
      N === 1 ? this.decodeScoresUniform : this.prefillScoresUniform,
      N === 1 ? this.decodeScoresBuf : undefined
    );
    this.pool.release(qRoped);

    // Softmax over scores
    const attnWeights = this.applySoftmax(
      encoder,
      scores,
      numAttentionHeads * N,
      totalSeq,
      N === 1 ? this.decodeSoftmaxUniform : this.prefillSoftmaxUniform,
      N === 1 ? this.decodeAttnWeightsBuf : undefined
    );
    if (N !== 1) this.pool.release(scores);

    // Attention output: weights @ V
    const attnOutput = this.computeAttnV(
      encoder,
      attnWeights,
      kvCache.value,
      N,
      totalSeq,
      N === 1 ? this.decodeAttnVUniform : this.prefillAttnVUniform
    );
    if (N !== 1) this.pool.release(attnWeights);

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
    posOffset: number,
    uniformBuf?: GPUBuffer,
    bgId?: string
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

    let paramsBuffer: GPUBuffer;
    if (uniformBuf) {
      this.device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(paramsData));
      paramsBuffer = uniformBuf;
    } else {
      paramsBuffer = this.createUniform(paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1 && bgId
      ? cachedBG(this.bgCache, this.device, bgId, bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

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
    S: number,
    uniformBuf?: GPUBuffer,
    preAllocated?: GPUBuffer
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "attention_scores",
      attentionWGSL,
      "compute_scores"
    );

    const { numAttentionHeads, numKeyValueHeads } = this.config;
    const scores = preAllocated ?? this.pool.acquire(
      numAttentionHeads * N * S * 4,
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

    let paramsBuffer: GPUBuffer;
    if (uniformBuf) {
      this.device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(paramsData));
      paramsBuffer = uniformBuf;
    } else {
      paramsBuffer = this.createUniform(paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: Q } },
      { binding: 1, resource: { buffer: K } },
      { binding: 2, resource: { buffer: scores } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1
      ? cachedBG(this.bgCache, this.device, "scores", bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

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
    D: number,
    uniformBuf?: GPUBuffer,
    preAllocated?: GPUBuffer
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "softmax",
      softmaxWGSL
    );

    const output = preAllocated ?? this.pool.acquire(
      N * D * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(8);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, D, true);

    let paramsBuffer: GPUBuffer;
    if (uniformBuf) {
      this.device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(paramsData));
      paramsBuffer = uniformBuf;
    } else {
      paramsBuffer = this.createUniform(paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1
      ? cachedBG(this.bgCache, this.device, "softmax", bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

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
    S: number,
    uniformBuf?: GPUBuffer
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

    let paramsBuffer: GPUBuffer;
    if (uniformBuf) {
      this.device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(paramsData));
      paramsBuffer = uniformBuf;
    } else {
      paramsBuffer = this.createUniform(paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: attn } },
      { binding: 1, resource: { buffer: V } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1
      ? cachedBG(this.bgCache, this.device, "attnV", bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

    const total = N * numAttentionHeads * this.hDim;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();

    return output;
  }

  /** Clear the bind group cache (call on KV cache reset). */
  clearBGCache(): void {
    clearBGCache(this.bgCache);
    this.qProj.clearBGCache();
    this.kProj.clearBGCache();
    this.vProj.clearBGCache();
    this.oProj.clearBGCache();
  }

  /** Destroy pre-allocated attention buffers. */
  destroyPreAllocated(): void {
    this.decodeScoresBuf?.destroy();
    this.decodeAttnWeightsBuf?.destroy();
    this.decodeScoresBuf = undefined;
    this.decodeAttnWeightsBuf = undefined;
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
