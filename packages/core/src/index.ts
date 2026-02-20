// 0xBitNet — WebGPU BitNet b1.58 Inference Engine

export type {
  ModelConfig,
  LoadOptions,
  LoadProgress,
  GenerateOptions,
  GPUContext,
  TokenizerConfig,
  ChatMessage,
  DiagnosticResult,
  WeightFormat,
  WorkerRequest,
  WorkerResponse,
} from "./types.js";

export { initGPU, GPUDeviceError } from "./gpu/device.js";
export { WeightStore } from "./model/weights.js";
export { loadModel, listCachedModels, deleteCachedModel } from "./model/loader.js";
export type { LoadResult } from "./model/loader.js";
export {
  BITNET_2B_4T_CONFIG,
  BITNET_0_7B_CONFIG,
  BITNET_3B_CONFIG,
} from "./model/config.js";
export { BitNetModel } from "./nn/model.js";
export { Tokenizer } from "./tokenizer/tokenizer.js";

import type { LoadOptions, GenerateOptions, ChatMessage } from "./types.js";
import { initGPU } from "./gpu/device.js";
import { loadModel } from "./model/loader.js";
import { BitNetModel } from "./nn/model.js";
import { Tokenizer } from "./tokenizer/tokenizer.js";
import type { DiagnosticResult } from "./types.js";

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
   * @param source - URL to a GGUF or Safetensors file.
   * @param options - Loading options including progress callback and abort signal.
   * @returns A ready-to-use `BitNet` instance.
   *
   * @example
   * ```ts
   * // Basic usage
   * const bitnet = await BitNet.load("https://example.com/model.gguf");
   *
   * // With progress and cancellation
   * const controller = new AbortController();
   * const bitnet = await BitNet.load(url, {
   *   onProgress: (p) => console.log(p.phase, p.fraction),
   *   signal: controller.signal,
   * });
   * ```
   */
  static async load(
    source: string | URL,
    options: LoadOptions = {}
  ): Promise<BitNet> {
    const gpu = options.device
      ? await initGPU(options.device)
      : await initGPU();

    const result = await loadModel(
      source,
      gpu.device,
      options.onProgress,
      options.signal
    );

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
      const tokResponse = await fetch(tokenizerUrl, { signal: options.signal });
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
   *
   * @param prompt - A plain text string or an array of chat messages.
   * @param options - Generation options including sampling parameters and abort signal.
   *
   * @example
   * ```ts
   * // Plain text prompt
   * for await (const token of bitnet.generate("Once upon a time")) {
   *   process.stdout.write(token);
   * }
   *
   * // Chat messages with cancellation
   * const controller = new AbortController();
   * const messages = [
   *   { role: "user" as const, content: "Summarize quantum computing" },
   * ];
   * for await (const token of bitnet.generate(messages, {
   *   maxTokens: 512,
   *   temperature: 0.7,
   *   signal: controller.signal,
   * })) {
   *   output += token;
   * }
   * ```
   */
  async *generate(
    prompt: string | ChatMessage[],
    options: GenerateOptions = {}
  ): AsyncGenerator<string> {
    const maxTokens = options.maxTokens ?? 256;
    const temperature = options.temperature ?? 1.0;
    const topK = options.topK ?? 50;
    const repeatPenalty = options.repeatPenalty ?? 1.0;
    const repeatLastN = options.repeatLastN ?? 64;
    const signal = options.signal;

    const inputIds = Array.isArray(prompt)
      ? this.tokenizer.applyChatTemplate(prompt)
      : this.tokenizer.encode(prompt);
    this.model.resetKVCache();

    const eotId = this.tokenizer.eotTokenId;
    const recentTokens: number[] = [];

    console.debug(`[0xBitNet] generate: ${inputIds.length} input tokens, eotId=${eotId}, temp=${temperature}, topK=${topK}, repeatPenalty=${repeatPenalty}`);
    console.debug(`[0xBitNet] first 20 token IDs:`, Array.from(inputIds.slice(0, 20)));

    // Prefill
    let logits = this.model.forward(inputIds);

    for (let i = 0; i < maxTokens; i++) {
      if (signal?.aborted) break;

      const window = repeatLastN > 0
        ? recentTokens.slice(-repeatLastN)
        : recentTokens;
      const nextToken = await this.sampleToken(
        logits,
        temperature,
        topK,
        repeatPenalty,
        window,
      );

      // Release the logits buffer back to the pool to prevent memory leak
      this.model.releaseBuffer(logits);

      if (nextToken === this.tokenizer.eosTokenId || nextToken === eotId) {
        break;
      }

      recentTokens.push(nextToken);
      const tokenStr = this.tokenizer.decode([nextToken]);
      options.onToken?.(tokenStr);
      yield tokenStr;

      // Decode step
      logits = this.model.forward(new Uint32Array([nextToken]));
    }
  }

  /**
   * Release all GPU resources held by this instance.
   * Must be called when the model is no longer needed.
   *
   * @example
   * ```ts
   * const bitnet = await BitNet.load(url);
   * // ... use the model ...
   * bitnet.dispose();
   * ```
   */
  dispose(): void {
    this.readbackBuffer.destroy();
    this.model.dispose();
  }

  /**
   * Run GPU diagnostic: forward pass stage-by-stage with readback.
   * Returns statistics for each intermediate tensor.
   *
   * @param prompt - Input text to run through the model (default: `"Hello"`).
   * @returns Array of diagnostic results, one per pipeline stage.
   *
   * @example
   * ```ts
   * const results = await bitnet.diagnose("Test input");
   * for (const r of results) {
   *   console.log(`${r.name}: mean=${r.mean.toFixed(4)}, rms=${r.rms.toFixed(4)}`);
   * }
   * ```
   */
  async diagnose(prompt = "Hello"): Promise<DiagnosticResult[]> {
    const inputIds = this.tokenizer.encode(prompt);
    return this.model.diagnose(inputIds);
  }

  private async sampleToken(
    logitsBuffer: GPUBuffer,
    temperature: number,
    topK: number,
    repeatPenalty: number,
    recentTokens: number[],
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

    // Repetition penalty (llama.cpp style)
    if (repeatPenalty !== 1.0 && recentTokens.length > 0) {
      for (const tokenId of recentTokens) {
        if (logits[tokenId] > 0) {
          logits[tokenId] /= repeatPenalty;
        } else {
          logits[tokenId] *= repeatPenalty;
        }
      }
    }

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
