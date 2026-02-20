// 0xBitNet — WebGPU BitNet b1.58 Inference Engine

export type {
  ModelConfig,
  LoadOptions,
  LoadProgress,
  GenerateOptions,
  GPUContext,
  KVCache,
  TokenizerConfig,
  WorkerRequest,
  WorkerResponse,
} from "./types.js";

export { initGPU, GPUDeviceError } from "./gpu/device.js";
export { PipelineManager } from "./gpu/pipeline.js";
export { BufferPool } from "./gpu/buffer-pool.js";
export { GGUFParser } from "./model/gguf.js";
export { parseSafetensorsHeader, getTensorInfos } from "./model/safetensors.js";
export { WeightStore } from "./model/weights.js";
export { loadModel } from "./model/loader.js";
export {
  BITNET_2B_4T_CONFIG,
  BITNET_0_7B_CONFIG,
  BITNET_3B_CONFIG,
} from "./model/config.js";
export { BitLinear } from "./nn/bitlinear.js";
export { Attention, createKVCache } from "./nn/attention.js";
export { FFN } from "./nn/ffn.js";
export { TransformerBlock } from "./nn/transformer.js";
export { BitNetModel } from "./nn/model.js";
export { Tokenizer } from "./tokenizer/tokenizer.js";

import type { LoadOptions, GenerateOptions, LoadProgress } from "./types.js";
import { initGPU } from "./gpu/device.js";
import { loadModel } from "./model/loader.js";
import { BitNetModel } from "./nn/model.js";
import { Tokenizer } from "./tokenizer/tokenizer.js";
import { GGUFParser } from "./model/gguf.js";

/**
 * High-level API for BitNet inference in the browser.
 *
 * @example
 * ```ts
 * const bitnet = await BitNet.load("https://example.com/model.gguf", {
 *   onProgress: (p) => console.log(`${(p.fraction * 100).toFixed(1)}%`),
 * });
 *
 * for await (const token of bitnet.generate("Hello, world!")) {
 *   process.stdout.write(token);
 * }
 *
 * bitnet.dispose();
 * ```
 */
export class BitNet {
  private model: BitNetModel;
  private tokenizer: Tokenizer;
  private device: GPUDevice;

  private constructor(
    model: BitNetModel,
    tokenizer: Tokenizer,
    device: GPUDevice
  ) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.device = device;
  }

  /**
   * Load a BitNet model from a URL.
   *
   * @param source URL to a GGUF or Safetensors file
   * @param options Loading options
   */
  static async load(
    source: string | URL,
    options: LoadOptions = {}
  ): Promise<BitNet> {
    const gpu = options.device
      ? await initGPU(options.device)
      : await initGPU();

    const result = await loadModel(source, gpu.device, options.onProgress);

    const model = BitNetModel.build(gpu.device, result.config, result.weights);

    // Extract tokenizer from GGUF if possible
    let tokenizer: Tokenizer;
    const url = typeof source === "string" ? source : source.href;
    if (url.endsWith(".gguf")) {
      // Re-parse just the header for tokenizer metadata
      const response = await fetch(url);
      const buffer = await response.arrayBuffer();
      const parser = new GGUFParser(buffer);
      const gguf = parser.parse();
      tokenizer = Tokenizer.fromGGUFMetadata(
        gguf.metadata as Record<string, unknown>
      );
    } else {
      // For safetensors, tokenizer must be loaded separately
      // Try to fetch tokenizer.json from the same directory
      const baseUrl = url.substring(0, url.lastIndexOf("/"));
      const tokenizerUrl = `${baseUrl}/tokenizer.json`;
      const tokResponse = await fetch(tokenizerUrl);
      if (tokResponse.ok) {
        const data = await tokResponse.json();
        tokenizer = Tokenizer.fromJSON(data);
      } else {
        throw new Error(
          "Could not find tokenizer. For safetensors models, " +
            "provide a tokenizer.json in the same directory."
        );
      }
    }

    return new BitNet(model, tokenizer, gpu.device);
  }

  /**
   * Generate text from a prompt.
   * Yields tokens as they are generated.
   */
  async *generate(
    prompt: string,
    options: GenerateOptions = {}
  ): AsyncGenerator<string> {
    const maxTokens = options.maxTokens ?? 256;
    const temperature = options.temperature ?? 1.0;
    const topK = options.topK ?? 50;

    const inputIds = this.tokenizer.encode(prompt);
    this.model.resetKVCache();

    // Prefill
    let logits = this.model.forward(inputIds);

    for (let i = 0; i < maxTokens; i++) {
      const nextToken = await this.sampleToken(
        logits,
        temperature,
        topK
      );

      // Release the logits buffer back to the pool to prevent memory leak
      this.model.releaseBuffer(logits);

      if (nextToken === this.tokenizer.eosTokenId) {
        break;
      }

      const tokenStr = this.tokenizer.decode([nextToken]);
      options.onToken?.(tokenStr);
      yield tokenStr;

      // Decode step
      logits = this.model.forward(new Uint32Array([nextToken]));
    }
  }

  /** Release all GPU resources. */
  dispose(): void {
    this.model.dispose();
  }

  private async sampleToken(
    logitsBuffer: GPUBuffer,
    temperature: number,
    topK: number
  ): Promise<number> {
    const vocabSize = this.model.config.vocabSize;

    const readBuffer = this.device.createBuffer({
      size: vocabSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    // model.forward() now always returns [1, V] logits (last token only)
    encoder.copyBufferToBuffer(
      logitsBuffer,
      0,
      readBuffer,
      0,
      vocabSize * 4
    );
    this.device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const logits = new Float32Array(
      readBuffer.getMappedRange().slice(0)
    );
    readBuffer.unmap();
    readBuffer.destroy();

    // Temperature
    if (temperature !== 1.0) {
      const invTemp = 1.0 / temperature;
      for (let i = 0; i < logits.length; i++) {
        logits[i] *= invTemp;
      }
    }

    // Top-K
    if (topK > 0 && topK < vocabSize) {
      const indices = Array.from({ length: vocabSize }, (_, i) => i);
      indices.sort((a, b) => logits[b] - logits[a]);
      const threshold = logits[indices[topK - 1]];
      for (let i = 0; i < vocabSize; i++) {
        if (logits[i] < threshold) {
          logits[i] = -Infinity;
        }
      }
    }

    // Softmax + sample
    let maxVal = -Infinity;
    for (const v of logits) {
      if (v > maxVal) maxVal = v;
    }
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      logits[i] = Math.exp(logits[i] - maxVal);
      sum += logits[i];
    }

    const r = Math.random() * sum;
    let cumsum = 0;
    for (let i = 0; i < logits.length; i++) {
      cumsum += logits[i];
      if (cumsum >= r) return i;
    }
    return logits.length - 1;
  }
}
