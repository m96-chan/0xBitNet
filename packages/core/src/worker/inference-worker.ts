import type { WorkerRequest, WorkerResponse, LoadProgress } from "../types.js";
import { initGPU } from "../gpu/device.js";
import { loadModel } from "../model/loader.js";
import { BitNetModel } from "../nn/model.js";
import { Tokenizer } from "../tokenizer/tokenizer.js";

let model: BitNetModel | null = null;
let tokenizer: Tokenizer | null = null;
let device: GPUDevice | null = null;

function postResponse(msg: WorkerResponse): void {
  self.postMessage(msg);
}

async function handleLoad(
  id: number,
  payload: { source: string }
): Promise<void> {
  try {
    const gpu = await initGPU();
    device = gpu.device;

    const result = await loadModel(
      payload.source,
      device,
      (progress: LoadProgress) => {
        postResponse({ id, type: "progress", payload: progress });
      }
    );

    model = BitNetModel.build(device, result.config, result.weights);

    // Try to extract tokenizer from GGUF metadata
    // This is set during loading if available
    // For now, the tokenizer must be loaded separately or extracted from GGUF

    postResponse({ id, type: "done" });
  } catch (err) {
    postResponse({
      id,
      type: "error",
      payload: { message: (err as Error).message },
    });
  }
}

async function handleGenerate(
  id: number,
  payload: {
    prompt: string | import("../types.js").ChatMessage[];
    options?: import("../types.js").GenerateOptions;
  }
): Promise<void> {
  if (!model || !tokenizer) {
    postResponse({
      id,
      type: "error",
      payload: { message: "Model not loaded" },
    });
    return;
  }

  try {
    const prompt = payload.prompt;
    const inputIds = Array.isArray(prompt)
      ? tokenizer.applyChatTemplate(prompt)
      : tokenizer.encode(prompt);
    const maxTokens = payload.options?.maxTokens ?? 256;
    const temperature = payload.options?.temperature ?? 1.0;
    const topK = payload.options?.topK ?? 50;

    model.resetKVCache();

    // Prefill: process full prompt
    let logits = model.forward(inputIds);

    for (let i = 0; i < maxTokens; i++) {
      // Sample from logits (last token position)
      const nextToken = await sampleFromLogits(
        logits,
        model.config.vocabSize,
        temperature,
        topK
      );

      if (nextToken === tokenizer.eosTokenId) {
        break;
      }

      const tokenStr = tokenizer.decode([nextToken]);
      postResponse({ id, type: "token", payload: tokenStr });

      // Decode step: process just the new token
      logits = model.forward(new Uint32Array([nextToken]));
    }

    postResponse({ id, type: "done" });
  } catch (err) {
    postResponse({
      id,
      type: "error",
      payload: { message: (err as Error).message },
    });
  }
}

function handleDispose(id: number): void {
  model?.dispose();
  model = null;
  tokenizer = null;
  postResponse({ id, type: "done" });
}

/**
 * Sample a token from logits on the GPU.
 * Reads back last-token logits, applies temperature + top-k sampling.
 */
async function sampleFromLogits(
  logitsBuffer: GPUBuffer,
  vocabSize: number,
  temperature: number,
  topK: number
): Promise<number> {
  if (!device) throw new Error("No GPU device");

  // Read back logits for the last token
  const readBuffer = device.createBuffer({
    size: vocabSize * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder();
  // Copy last row of logits
  const srcOffset =
    (logitsBuffer.size / 4 / vocabSize - 1) * vocabSize * 4;
  encoder.copyBufferToBuffer(
    logitsBuffer,
    Math.max(0, srcOffset),
    readBuffer,
    0,
    vocabSize * 4
  );
  device.queue.submit([encoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const logits = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  readBuffer.destroy();

  // Apply temperature
  if (temperature !== 1.0) {
    const invTemp = 1.0 / temperature;
    for (let i = 0; i < logits.length; i++) {
      logits[i] *= invTemp;
    }
  }

  // Top-K filtering
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

  // Softmax
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    logits[i] = Math.exp(logits[i] - maxVal);
    sum += logits[i];
  }
  for (let i = 0; i < logits.length; i++) {
    logits[i] /= sum;
  }

  // Multinomial sampling
  const r = Math.random();
  let cumsum = 0;
  for (let i = 0; i < logits.length; i++) {
    cumsum += logits[i];
    if (cumsum >= r) {
      return i;
    }
  }

  return logits.length - 1;
}

// Message handler
self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  switch (msg.type) {
    case "load":
      handleLoad(msg.id, msg.payload);
      break;
    case "generate":
      handleGenerate(msg.id, msg.payload);
      break;
    case "dispose":
      handleDispose(msg.id);
      break;
  }
});
