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
export { loadModel, listCachedModels, deleteCachedModel } from "./model/loader.js";
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
export type { DiagnosticResult } from "./nn/model.js";
export { Tokenizer } from "./tokenizer/tokenizer.js";
export type { ChatMessage } from "./tokenizer/tokenizer.js";

import type { LoadOptions, GenerateOptions, LoadProgress } from "./types.js";
import { initGPU } from "./gpu/device.js";
import { loadModel } from "./model/loader.js";
import { BitNetModel } from "./nn/model.js";
import { Tokenizer } from "./tokenizer/tokenizer.js";
import type { ChatMessage } from "./tokenizer/tokenizer.js";

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
  private readbackBuffer: GPUBuffer;
  private logitsArray: Float32Array;

  private constructor(
    model: BitNetModel,
    tokenizer: Tokenizer,
    device: GPUDevice
  ) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.device = device;
    const vocabSize = model.config.vocabSize;
    this.readbackBuffer = device.createBuffer({
      size: vocabSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.logitsArray = new Float32Array(vocabSize);
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

    // Extract tokenizer from GGUF metadata or fetch tokenizer.json
    let tokenizer: Tokenizer;
    const url = typeof source === "string" ? source : source.href;
    if (url.endsWith(".gguf") && result.metadata) {
      tokenizer = Tokenizer.fromGGUFMetadata(result.metadata);
    } else if (!url.endsWith(".gguf")) {
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
    } else {
      throw new Error(
        "Could not extract tokenizer metadata from GGUF file."
      );
    }

    return new BitNet(model, tokenizer, gpu.device);
  }

  /**
   * Generate text from a prompt.
   * Yields tokens as they are generated.
   */
  async *generate(
    prompt: string | ChatMessage[],
    options: GenerateOptions = {}
  ): AsyncGenerator<string> {
    const maxTokens = options.maxTokens ?? 256;
    const temperature = options.temperature ?? 1.0;
    const topK = options.topK ?? 50;

    const inputIds = Array.isArray(prompt)
      ? this.tokenizer.applyChatTemplate(prompt)
      : this.tokenizer.encode(prompt);
    this.model.resetKVCache();

    const eotId = this.tokenizer.eotTokenId;

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

      if (nextToken === this.tokenizer.eosTokenId || nextToken === eotId) {
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
    this.readbackBuffer.destroy();
    this.model.dispose();
  }

  /**
   * Run GPU diagnostic: forward pass stage-by-stage with readback.
   * Returns statistics for each intermediate tensor.
   */
  async diagnose(prompt = "Hello"): Promise<import("./nn/model.js").DiagnosticResult[]> {
    const inputIds = this.tokenizer.encode(prompt);
    return this.model.diagnose(inputIds);
  }

  private async sampleToken(
    logitsBuffer: GPUBuffer,
    temperature: number,
    topK: number
  ): Promise<number> {
    const vocabSize = this.model.config.vocabSize;

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      logitsBuffer,
      0,
      this.readbackBuffer,
      0,
      vocabSize * 4
    );
    this.device.queue.submit([encoder.finish()]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const mapped = new Float32Array(this.readbackBuffer.getMappedRange());
    const logits = this.logitsArray;
    logits.set(mapped);
    this.readbackBuffer.unmap();

    // Temperature
    if (temperature !== 1.0) {
      const invTemp = 1.0 / temperature;
      for (let i = 0; i < vocabSize; i++) {
        logits[i] *= invTemp;
      }
    }

    // Top-K via min-heap (O(V) instead of O(V log V) sort)
    if (topK > 0 && topK < vocabSize) {
      const heap = new Uint32Array(topK);
      for (let i = 0; i < topK; i++) heap[i] = i;
      // Build initial min-heap
      for (let i = (topK >> 1) - 1; i >= 0; i--) {
        siftDown(heap, i, topK, logits);
      }
      // Process remaining elements
      for (let i = topK; i < vocabSize; i++) {
        if (logits[i] > logits[heap[0]]) {
          heap[0] = i;
          siftDown(heap, 0, topK, logits);
        }
      }
      const threshold = logits[heap[0]];
      for (let i = 0; i < vocabSize; i++) {
        if (logits[i] < threshold) logits[i] = -Infinity;
      }
    }

    // Softmax + sample
    let maxVal = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      if (logits[i] > maxVal) maxVal = logits[i];
    }
    let sum = 0;
    for (let i = 0; i < vocabSize; i++) {
      logits[i] = Math.exp(logits[i] - maxVal);
      sum += logits[i];
    }

    const r = Math.random() * sum;
    let cumsum = 0;
    for (let i = 0; i < vocabSize; i++) {
      cumsum += logits[i];
      if (cumsum >= r) return i;
    }
    return vocabSize - 1;
  }
}

/** Min-heap sift-down for top-K selection */
function siftDown(
  heap: Uint32Array,
  i: number,
  n: number,
  values: Float32Array
): void {
  while (true) {
    let min = i;
    const l = 2 * i + 1;
    const r = 2 * i + 2;
    if (l < n && values[heap[l]] < values[heap[min]]) min = l;
    if (r < n && values[heap[r]] < values[heap[min]]) min = r;
    if (min === i) break;
    const tmp = heap[i];
    heap[i] = heap[min];
    heap[min] = tmp;
    i = min;
  }
}
