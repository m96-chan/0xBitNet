import { PipelineManager } from "../gpu/pipeline.js";
import { BufferPool } from "../gpu/buffer-pool.js";
import { BitLinear } from "./bitlinear.js";
import type { ModelConfig } from "../types.js";

import activationWGSL from "../shaders/activation.wgsl";
import elementwiseWGSL from "../shaders/elementwise.wgsl";

/**
 * Feed-Forward Network with two variants:
 *
 * 1. ReLU² (official 2B-4T model):
 *    out = down_proj(relu²(up_proj(x)))
 *
 * 2. SwiGLU (community models):
 *    out = down_proj(silu(gate_proj(x)) * up_proj(x))
 */
export class FFN {
  private device: GPUDevice;
  private pipelines: PipelineManager;
  private pool: BufferPool;
  private config: ModelConfig;

  private upProj: BitLinear;
  private downProj: BitLinear;
  private gateProj: BitLinear | null; // null for ReLU² (no gate)

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

  /**
   * Forward pass: input [N, hidden] → output [N, hidden]
   */
  forward(
    input: GPUBuffer,
    N: number,
    encoder: GPUCommandEncoder
  ): GPUBuffer {
    if (this.config.activation === "relu2") {
      return this.forwardReLU2(input, N, encoder);
    } else {
      return this.forwardSwiGLU(input, N, encoder);
    }
  }

  private forwardReLU2(
    input: GPUBuffer,
    N: number,
    encoder: GPUCommandEncoder
  ): GPUBuffer {
    // up = up_proj(x)
    const up = this.upProj.forward(input, N, encoder);

    // activated = relu²(up)
    const activated = this.applyActivation(
      encoder,
      up,
      N * this.config.intermediateSize,
      0 // ReLU²
    );
    this.pool.release(up);

    // out = down_proj(activated)
    const output = this.downProj.forward(activated, N, encoder);
    this.pool.release(activated);

    return output;
  }

  private forwardSwiGLU(
    input: GPUBuffer,
    N: number,
    encoder: GPUCommandEncoder
  ): GPUBuffer {
    // gate = gate_proj(x)
    const gate = this.gateProj!.forward(input, N, encoder);

    // up = up_proj(x)
    const up = this.upProj.forward(input, N, encoder);

    // gate_activated = silu(gate)
    const gateActivated = this.applyActivation(
      encoder,
      gate,
      N * this.config.intermediateSize,
      1 // SiLU
    );
    this.pool.release(gate);

    // gated = silu(gate) * up
    const gated = this.applyElementwise(
      encoder,
      gateActivated,
      up,
      N * this.config.intermediateSize,
      1 // multiply
    );
    this.pool.release(gateActivated);
    this.pool.release(up);

    // out = down_proj(gated)
    const output = this.downProj.forward(gated, N, encoder);
    this.pool.release(gated);

    return output;
  }

  private applyActivation(
    encoder: GPUCommandEncoder,
    input: GPUBuffer,
    numElements: number,
    activationType: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      `activation_${activationType}`,
      activationWGSL
    );

    const output = this.pool.acquire(
      numElements * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(8);
    const v = new DataView(paramsData);
    v.setUint32(0, numElements, true);
    v.setUint32(4, activationType, true);
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
    pass.dispatchWorkgroups(Math.ceil(numElements / 256));
    pass.end();

    return output;
  }

  private applyElementwise(
    encoder: GPUCommandEncoder,
    a: GPUBuffer,
    b: GPUBuffer,
    numElements: number,
    op: number
  ): GPUBuffer {
    const { pipeline, bindGroupLayout } = this.pipelines.getOrCreate(
      `elementwise_${op}`,
      elementwiseWGSL
    );

    const output = this.pool.acquire(
      numElements * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const paramsData = new ArrayBuffer(8);
    const v = new DataView(paramsData);
    v.setUint32(0, numElements, true);
    v.setUint32(4, op, true);
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
