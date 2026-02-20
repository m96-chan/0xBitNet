import { PipelineManager } from "../gpu/pipeline.js";
import { BufferPool } from "../gpu/buffer-pool.js";
import { TransformerBlock } from "./transformer.js";
import { BitLinear } from "./bitlinear.js";
import { Attention, createKVCache } from "./attention.js";
import { FFN } from "./ffn.js";
import { WeightStore } from "../model/weights.js";
import type { ModelConfig, KVCache } from "../types.js";
import { headDim } from "../model/config.js";

import embeddingWGSL from "../shaders/embedding.wgsl";
import rmsnormWGSL from "../shaders/rmsnorm.wgsl";
import f32MatmulWGSL from "../shaders/f32_matmul.wgsl";

/**
 * Full BitNet model: embedding → N × transformer → final RMSNorm → LM head
 */
export class BitNetModel {
  private device: GPUDevice;
  private pipelines: PipelineManager;
  private pool: BufferPool;
  config: ModelConfig;

  private embedTokens: GPUBuffer;
  private layers: TransformerBlock[];
  private finalNorm: GPUBuffer;
  private lmHead: BitLinear | GPUBuffer; // BitLinear or tied embedding

  private kvCaches: KVCache[];

  constructor(
    device: GPUDevice,
    pipelines: PipelineManager,
    pool: BufferPool,
    config: ModelConfig,
    embedTokens: GPUBuffer,
    layers: TransformerBlock[],
    finalNorm: GPUBuffer,
    lmHead: BitLinear | GPUBuffer,
    kvCaches: KVCache[]
  ) {
    this.device = device;
    this.pipelines = pipelines;
    this.pool = pool;
    this.config = config;
    this.embedTokens = embedTokens;
    this.layers = layers;
    this.finalNorm = finalNorm;
    this.lmHead = lmHead;
    this.kvCaches = kvCaches;
  }

  /**
   * Build a full model from loaded weights.
   */
  static build(
    device: GPUDevice,
    config: ModelConfig,
    weights: WeightStore,
    maxSeqLen = 4096
  ): BitNetModel {
    const pipelines = new PipelineManager(device);
    const pool = new BufferPool(device);

    // Helper to get a weight buffer with a clear error on missing
    function requireWeight(name: string): GPUBuffer {
      const buf = weights.get(name);
      if (!buf) {
        throw new Error(`Missing weight tensor: "${name}"`);
      }
      return buf;
    }

    const embedTokens = requireWeight("model.embed_tokens.weight");
    const finalNorm = requireWeight("model.norm.weight");

    const layers: TransformerBlock[] = [];
    const kvCaches: KVCache[] = [];

    for (let i = 0; i < config.numHiddenLayers; i++) {
      const prefix = `model.layers.${i}`;

      const inputLN = requireWeight(`${prefix}.input_layernorm.weight`);
      const postAttnLN = requireWeight(`${prefix}.post_attention_layernorm.weight`);

      // BitNet sub-norms: applied at specific positions per reference model
      // attn_sub_norm (dim=hiddenSize): applied before o_proj only
      // ffn_sub_norm (dim=intermediateSize): applied before down_proj only
      const attnSubNorm = weights.get(`${prefix}.self_attn.sub_norm.weight`) ?? null;
      const ffnSubNorm = weights.get(`${prefix}.mlp.sub_norm.weight`) ?? null;

      // Q/K/V: no sub-norm (already normalized by input_layernorm in TransformerBlock)
      const qProj = new BitLinear(
        device, pipelines, pool,
        requireWeight(`${prefix}.self_attn.q_proj.weight`),
        requireWeight(`${prefix}.self_attn.q_proj.weight_scale`),
        null,
        config.hiddenSize,
        config.numAttentionHeads * headDim(config)
      );

      const kProj = new BitLinear(
        device, pipelines, pool,
        requireWeight(`${prefix}.self_attn.k_proj.weight`),
        requireWeight(`${prefix}.self_attn.k_proj.weight_scale`),
        null,
        config.hiddenSize,
        config.numKeyValueHeads * headDim(config)
      );

      const vProj = new BitLinear(
        device, pipelines, pool,
        requireWeight(`${prefix}.self_attn.v_proj.weight`),
        requireWeight(`${prefix}.self_attn.v_proj.weight_scale`),
        null,
        config.hiddenSize,
        config.numKeyValueHeads * headDim(config)
      );

      // O: uses attn_sub_norm (dim=hiddenSize, applied after attention output)
      const oProj = new BitLinear(
        device, pipelines, pool,
        requireWeight(`${prefix}.self_attn.o_proj.weight`),
        requireWeight(`${prefix}.self_attn.o_proj.weight_scale`),
        attnSubNorm,
        config.numAttentionHeads * headDim(config),
        config.hiddenSize
      );

      const attention = new Attention(
        device, pipelines, pool, config,
        qProj, kProj, vProj, oProj
      );

      // FFN projections
      // up_proj: no sub-norm (already normalized by post_attention_layernorm)
      const upProj = new BitLinear(
        device, pipelines, pool,
        requireWeight(`${prefix}.mlp.up_proj.weight`),
        requireWeight(`${prefix}.mlp.up_proj.weight_scale`),
        null,
        config.hiddenSize,
        config.intermediateSize
      );

      // down_proj: uses ffn_sub_norm (dim=intermediateSize)
      const downProj = new BitLinear(
        device, pipelines, pool,
        requireWeight(`${prefix}.mlp.down_proj.weight`),
        requireWeight(`${prefix}.mlp.down_proj.weight_scale`),
        ffnSubNorm,
        config.intermediateSize,
        config.hiddenSize
      );

      // gate_proj: create if weights exist (2B-4T uses gated relu²)
      let gateProj: BitLinear | null = null;
      if (weights.has(`${prefix}.mlp.gate_proj.weight`)) {
        gateProj = new BitLinear(
          device, pipelines, pool,
          requireWeight(`${prefix}.mlp.gate_proj.weight`),
          requireWeight(`${prefix}.mlp.gate_proj.weight_scale`),
          null,
          config.hiddenSize,
          config.intermediateSize
        );
      }

      const ffn = new FFN(
        device, pipelines, pool, config,
        upProj, downProj, gateProj
      );

      layers.push(
        new TransformerBlock(
          device,
          pipelines,
          pool,
          config,
          inputLN,
          postAttnLN,
          attention,
          ffn
        )
      );

      kvCaches.push(createKVCache(device, config, maxSeqLen));
    }

    // LM head: tied to embedding if no separate weight exists
    let lmHead: BitLinear | GPUBuffer;
    if (config.tieWordEmbeddings || !weights.has("lm_head.weight")) {
      lmHead = embedTokens;
    } else {
      lmHead = new BitLinear(
        device,
        pipelines,
        pool,
        requireWeight("lm_head.weight"),
        requireWeight("lm_head.weight_scale"),
        weights.get("lm_head.input_norm.weight") ?? finalNorm,
        config.hiddenSize,
        config.vocabSize
      );
    }

    return new BitNetModel(
      device,
      pipelines,
      pool,
      config,
      embedTokens,
      layers,
      finalNorm,
      lmHead,
      kvCaches
    );
  }

  /**
   * Forward pass: token IDs → logits
   * @param tokenIds Array of token IDs
   * @returns logits buffer [N, vocabSize] f32
   */
  forward(tokenIds: Uint32Array): GPUBuffer {
    const N = tokenIds.length;
    const encoder = this.device.createCommandEncoder();

    // Upload token IDs
    const tokenBuffer = this.device.createBuffer({
      size: tokenIds.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(tokenBuffer.getMappedRange()).set(tokenIds);
    tokenBuffer.unmap();

    // Embedding lookup
    let hidden = this.dispatchEmbedding(encoder, tokenBuffer, N);

    // Transformer layers
    for (let i = 0; i < this.layers.length; i++) {
      const newHidden = this.layers[i].forward(
        hidden,
        N,
        this.kvCaches[i],
        encoder
      );
      this.pool.release(hidden);
      hidden = newHidden;

      // Update KV cache sequence length after processing
      this.kvCaches[i].seqLen += N;
    }

    // Final RMSNorm
    const normed = this.dispatchFinalNorm(encoder, hidden, N);
    this.pool.release(hidden);

    // LM head
    let logits: GPUBuffer;
    if (this.lmHead instanceof BitLinear) {
      logits = (this.lmHead as BitLinear).forward(normed, N, encoder);
    } else {
      // Tied embedding: simple matmul
      logits = this.dispatchLMHead(encoder, normed, N);
    }
    this.pool.release(normed);

    this.device.queue.submit([encoder.finish()]);
    return logits;
  }

  /** Reset all KV caches (for new generation) */
  resetKVCache(): void {
    for (const cache of this.kvCaches) {
      cache.seqLen = 0;
    }
  }

  /** Destroy all resources */
  dispose(): void {
    for (const cache of this.kvCaches) {
      cache.key.destroy();
      cache.value.destroy();
    }
    this.pool.destroy();
    this.pipelines.clear();
  }

  private dispatchEmbedding(
    encoder: GPUCommandEncoder,
    tokenBuffer: GPUBuffer,
    N: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "embedding",
      embeddingWGSL
    );

    const outputSize = N * this.config.hiddenSize * 4;
    const output = this.pool.acquire(
      outputSize,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(12);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, this.config.hiddenSize, true);
    v.setUint32(8, this.config.vocabSize, true);
    const paramsBuffer = this.createUniform(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: tokenBuffer } },
        { binding: 1, resource: { buffer: this.embedTokens } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const total = N * this.config.hiddenSize;
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();

    return output;
  }

  private dispatchFinalNorm(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    N: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "rmsnorm",
      rmsnormWGSL
    );

    const output = this.pool.acquire(
      N * this.config.hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(12);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, this.config.hiddenSize, true);
    v.setFloat32(8, this.config.rmsNormEps, true);
    const paramsBuffer = this.createUniform(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: this.finalNorm } },
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

  private dispatchLMHead(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    N: number
  ): GPUBuffer {
    const V = this.config.vocabSize;
    const D = this.config.hiddenSize;

    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      "f32_matmul",
      f32MatmulWGSL
    );

    const output = this.pool.acquire(
      N * V * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(12);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, V, true);
    v.setUint32(8, D, true);
    const paramsBuffer = this.createUniform(paramsData);

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: this.embedTokens } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const totalWorkgroups = N * V;
    const wgX = Math.min(totalWorkgroups, 65535);
    const wgY = Math.ceil(totalWorkgroups / 65535);

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY);
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
