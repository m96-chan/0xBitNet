import { PipelineManager } from "../gpu/pipeline.js";
import { BufferPool } from "../gpu/buffer-pool.js";
import { BitLinear } from "./bitlinear.js";
import { type BindGroupCache, createBGCache, clearBGCache, cachedBG } from "./bg-cache.js";
import type { ModelConfig } from "../types.js";

import activationWGSL from "../shaders/activation.wgsl";
import elementwiseWGSL from "../shaders/elementwise.wgsl";

/**
 * Feed-Forward Network with two variants:
 *
 * 1. Gated ReLU² (official 2B-4T model):
 *    out = down_proj(ffn_sub_norm(relu²(gate_proj(x)) * up_proj(x)))
 *
 * 2. SwiGLU (community models):
 *    out = down_proj(silu(gate_proj(x)) * up_proj(x))
 *
 * Both use gating. The sub-norm before down_proj is handled inside
 * down_proj's BitLinear (which has ffn_sub_norm as its normWeight).
 */
export class FFN {
  private device: GPUDevice;
  private pipelines: PipelineManager;
  private pool: BufferPool;
  private config: ModelConfig;

  private upProj: BitLinear;
  private downProj: BitLinear;
  private gateProj: BitLinear | null; // null for ReLU² (no gate)

  // Pre-created uniform buffers for N=1 decode (static params)
  private decodeActivationUniform?: GPUBuffer;
  private decodeElementwiseUniform?: GPUBuffer;

  // Pre-created uniform buffers for N>1 prefill (dynamic — updated via writeBuffer)
  private prefillActivationUniform?: GPUBuffer;
  private prefillElementwiseUniform?: GPUBuffer;

  // Bind group cache for N=1 decode
  private bgCache: BindGroupCache = createBGCache();

  constructor(
    device: GPUDevice,
    pipelines: PipelineManager,
    pool: BufferPool,
    config: ModelConfig,
    upProj: BitLinear,
    downProj: BitLinear,
    gateProj: BitLinear | null
  ) {
    this.device = device;
    this.pipelines = pipelines;
    this.pool = pool;
    this.config = config;
    this.upProj = upProj;
    this.downProj = downProj;
    this.gateProj = gateProj;
  }

  /** Pre-create uniform buffers for N=1 decode path (all static). */
  initDecodeUniforms(): void {
    const activationType = this.config.activation === "relu2" ? 0 : 1;
    {
      const data = new ArrayBuffer(8);
      const v = new DataView(data);
      v.setUint32(0, this.config.intermediateSize, true);
      v.setUint32(4, activationType, true);
      this.decodeActivationUniform = this.createUniform(data);
    }
    {
      const data = new ArrayBuffer(8);
      const v = new DataView(data);
      v.setUint32(0, this.config.intermediateSize, true);
      v.setUint32(4, 1, true); // multiply
      this.decodeElementwiseUniform = this.createUniform(data);
    }

    // Prefill uniforms (reused via writeBuffer for N>1)
    const mkBuf = (size: number) =>
      this.device.createBuffer({
        size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.prefillActivationUniform = mkBuf(8);
    this.prefillElementwiseUniform = mkBuf(8);

    this.upProj.initDecodeUniforms();
    this.downProj.initDecodeUniforms();
    this.gateProj?.initDecodeUniforms();
  }

  /**
   * Forward pass: input [N, hidden] → output [N, hidden]
   */
  forward(
    input: GPUBuffer,
    N: number,
    encoder: GPUCommandEncoder
  ): GPUBuffer {
    if (this.gateProj) {
      // Gated forward: activation(gate(x)) * up(x)
      // Used by both relu² (2B-4T) and SwiGLU (community models)
      return this.forwardGated(input, N, encoder);
    } else {
      // Simple forward: activation(up(x)) — fallback if no gate weights
      return this.forwardSimple(input, N, encoder);
    }
  }

  private forwardGated(
    input: GPUBuffer,
    N: number,
    encoder: GPUCommandEncoder
  ): GPUBuffer {
    const activationType = this.config.activation === "relu2" ? 0 : 1;

    // gate = gate_proj(x), up = up_proj(x)
    const gate = this.gateProj!.forward(input, N, encoder);
    const up = this.upProj.forward(input, N, encoder);

    // gate_activated = activation(gate)  — relu² or SiLU
    const gateActivated = this.applyActivation(
      encoder,
      gate,
      N * this.config.intermediateSize,
      activationType,
      N
    );
    this.pool.release(gate);

    // gated = activation(gate) * up
    const gated = this.applyElementwise(
      encoder,
      gateActivated,
      up,
      N * this.config.intermediateSize,
      1, // multiply
      N
    );
    this.pool.release(gateActivated);
    this.pool.release(up);

    // out = down_proj(gated)
    // down_proj's BitLinear applies ffn_sub_norm internally before quantization
    const output = this.downProj.forward(gated, N, encoder);
    this.pool.release(gated);

    return output;
  }

  private forwardSimple(
    input: GPUBuffer,
    N: number,
    encoder: GPUCommandEncoder
  ): GPUBuffer {
    const activationType = this.config.activation === "relu2" ? 0 : 1;

    // up = up_proj(x)
    const up = this.upProj.forward(input, N, encoder);

    // activated = activation(up)
    const activated = this.applyActivation(
      encoder,
      up,
      N * this.config.intermediateSize,
      activationType,
      N
    );
    this.pool.release(up);

    // out = down_proj(activated)
    const output = this.downProj.forward(activated, N, encoder);
    this.pool.release(activated);

    return output;
  }

  private applyActivation(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    numElements: number,
    activationType: number,
    N: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      `activation_${activationType}`,
      activationWGSL
    );

    const output = this.pool.acquire(
      numElements * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    let paramsBuffer: GPUBuffer;
    if (N === 1 && this.decodeActivationUniform) {
      paramsBuffer = this.decodeActivationUniform;
    } else if (this.prefillActivationUniform) {
      const paramsData = new ArrayBuffer(8);
      const v = new DataView(paramsData);
      v.setUint32(0, numElements, true);
      v.setUint32(4, activationType, true);
      this.device.queue.writeBuffer(this.prefillActivationUniform, 0, new Uint8Array(paramsData));
      paramsBuffer = this.prefillActivationUniform;
    } else {
      const paramsData = new ArrayBuffer(8);
      const v = new DataView(paramsData);
      v.setUint32(0, numElements, true);
      v.setUint32(4, activationType, true);
      paramsBuffer = this.createUniform(paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1
      ? cachedBG(this.bgCache, this.device, "activation", bindGroupLayout, entries)
      : this.device.createBindGroup({ layout: bindGroupLayout, entries });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(numElements / 256));
    pass.end();

    return output;
  }

  private applyElementwise(
    encoder: GPUCommandEncoder,
    a: GPUBuffer,
    b: GPUBuffer,
    numElements: number,
    op: number,
    N: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      `elementwise_${op}`,
      elementwiseWGSL
    );

    const output = this.pool.acquire(
      numElements * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    let paramsBuffer: GPUBuffer;
    if (N === 1 && this.decodeElementwiseUniform) {
      paramsBuffer = this.decodeElementwiseUniform;
    } else if (this.prefillElementwiseUniform) {
      const paramsData = new ArrayBuffer(8);
      const v = new DataView(paramsData);
      v.setUint32(0, numElements, true);
      v.setUint32(4, op, true);
      this.device.queue.writeBuffer(this.prefillElementwiseUniform, 0, new Uint8Array(paramsData));
      paramsBuffer = this.prefillElementwiseUniform;
    } else {
      const paramsData = new ArrayBuffer(8);
      const v = new DataView(paramsData);
      v.setUint32(0, numElements, true);
      v.setUint32(4, op, true);
      paramsBuffer = this.createUniform(paramsData);
    }

    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: a } },
      { binding: 1, resource: { buffer: b } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ];
    const bindGroup = N === 1
      ? cachedBG(this.bgCache, this.device, "elementwise", bindGroupLayout, entries)
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
    this.upProj.clearBGCache();
    this.downProj.clearBGCache();
    this.gateProj?.clearBGCache();
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
