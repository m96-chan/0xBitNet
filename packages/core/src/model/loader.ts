import type { LoadProgress, ModelConfig, WeightFormat } from "../types.js";
import { GGUFParser, ggmlTypeSize, GGML_TYPE_I2_S } from "./gguf.js";
import { parseSafetensorsHeader, getTensorInfos } from "./safetensors.js";
import {
  BITNET_2B_4T_CONFIG,
  BITNET_0_7B_CONFIG,
  BITNET_3B_CONFIG,
} from "./config.js";
import { WeightStore } from "./weights.js";

export interface LoadResult {
  config: ModelConfig;
  weights: WeightStore;
}

/**
 * Fetch a model from a URL, cache it, and upload weights to GPU.
 *
 * Supports:
 * - GGUF files (auto-detected by magic bytes or .gguf extension)
 * - Safetensors files (.safetensors extension)
 * - Hugging Face model repos (auto-discovers format)
 */
export async function loadModel(
  source: string | URL,
  device: GPUDevice,
  onProgress?: (progress: LoadProgress) => void
): Promise<LoadResult> {
  const url = typeof source === "string" ? source : source.href;
  const format = detectFormat(url);

  onProgress?.({ phase: "download", loaded: 0, total: 0, fraction: 0 });

  const buffer = await fetchModel(url, (loaded, total) => {
    onProgress?.({
      phase: "download",
      loaded,
      total,
      fraction: total > 0 ? loaded / total : 0,
    });
  });

  onProgress?.({ phase: "parse", loaded: 0, total: 1, fraction: 0 });

  if (format === "gguf") {
    return loadGGUF(buffer, device, onProgress);
  } else {
    return loadSafetensors(buffer, device, onProgress);
  }
}

function detectFormat(url: string): WeightFormat {
  if (url.endsWith(".gguf")) return "gguf";
  if (url.endsWith(".safetensors")) return "safetensors";
  // Default to GGUF
  return "gguf";
}

async function loadGGUF(
  buffer: ArrayBuffer,
  device: GPUDevice,
  onProgress?: (progress: LoadProgress) => void
): Promise<LoadResult> {
  const parser = new GGUFParser(buffer);
  const gguf = parser.parse();

  // Derive config from metadata
  const config = configFromGGUFMetadata(gguf.metadata);

  const store = new WeightStore(device);
  const maxBinding = device.limits.maxStorageBufferBindingSize;
  const totalTensors = gguf.tensors.length;

  for (let i = 0; i < totalTensors; i++) {
    const tensor = gguf.tensors[i];
    const dataOffset = gguf.tensorDataOffset + Number(tensor.offset);

    // Calculate tensor byte size
    const numElements = tensor.shape.reduce(
      (a, b) => a * Number(b),
      1
    );
    const elemSize = ggmlTypeSize(tensor.type);
    const byteSize = Math.ceil(numElements * elemSize);
    const tensorData = buffer.slice(dataOffset, dataOffset + byteSize);

    store.uploadSharded(tensor.name, tensorData, maxBinding);

    onProgress?.({
      phase: "upload",
      loaded: i + 1,
      total: totalTensors,
      fraction: (i + 1) / totalTensors,
    });
  }

  return { config, weights: store };
}

async function loadSafetensors(
  buffer: ArrayBuffer,
  device: GPUDevice,
  onProgress?: (progress: LoadProgress) => void
): Promise<LoadResult> {
  const { header, dataOffset } = parseSafetensorsHeader(buffer);
  const tensorInfos = getTensorInfos(header, dataOffset);

  // Infer config from tensor shapes
  const config = configFromSafetensors(tensorInfos);

  const store = new WeightStore(device);
  const maxBinding = device.limits.maxStorageBufferBindingSize;

  for (let i = 0; i < tensorInfos.length; i++) {
    const info = tensorInfos[i];
    const tensorData = buffer.slice(info.offset, info.offset + info.size);
    store.uploadSharded(info.name, tensorData, maxBinding);

    onProgress?.({
      phase: "upload",
      loaded: i + 1,
      total: tensorInfos.length,
      fraction: (i + 1) / tensorInfos.length,
    });
  }

  return { config, weights: store };
}

function configFromGGUFMetadata(
  metadata: Record<string, unknown>
): ModelConfig {
  const get = (key: string) => metadata[key];

  const hiddenSize =
    (get("llama.embedding_length") as number) ??
    (get("bitnet.embedding_length") as number) ??
    2560;
  const numLayers =
    (get("llama.block_count") as number) ??
    (get("bitnet.block_count") as number) ??
    30;
  const numHeads =
    (get("llama.attention.head_count") as number) ??
    (get("bitnet.attention.head_count") as number) ??
    32;
  const numKVHeads =
    (get("llama.attention.head_count_kv") as number) ??
    (get("bitnet.attention.head_count_kv") as number) ??
    numHeads;
  const vocabSize =
    (get("llama.vocab_size") as number) ??
    (get("tokenizer.ggml.tokens") as unknown[] as string[])?.length ??
    128256;
  const intermediateSize =
    (get("llama.feed_forward_length") as number) ??
    (get("bitnet.feed_forward_length") as number) ??
    6912;

  // Try to detect activation type from architecture
  const arch =
    (get("general.architecture") as string) ?? "bitnet";
  const isOfficial =
    vocabSize > 100000 || arch.includes("bitnet");

  return {
    modelType: "bitnet",
    vocabSize,
    hiddenSize,
    intermediateSize,
    numHiddenLayers: numLayers,
    numAttentionHeads: numHeads,
    numKeyValueHeads: numKVHeads,
    maxPositionEmbeddings: (get("llama.context_length") as number) ?? 4096,
    rmsNormEps: (get("llama.attention.layer_norm_rms_epsilon") as number) ?? 1e-5,
    ropeTheta:
      (get("llama.rope.freq_base") as number) ??
      (isOfficial ? 500000.0 : 10000.0),
    tieWordEmbeddings: false,
    activation: isOfficial ? "relu2" : "silu",
  };
}

function configFromSafetensors(
  tensors: { name: string; shape: number[] }[]
): ModelConfig {
  // Infer dimensions from weight shapes
  const embedTensor = tensors.find(
    (t) =>
      t.name === "model.embed_tokens.weight" ||
      t.name === "transformer.wte.weight"
  );
  const vocabSize = embedTensor?.shape[0] ?? 128256;
  const hiddenSize = embedTensor?.shape[1] ?? 2560;

  // Count layers
  const layerIndices = tensors
    .map((t) => {
      const m = t.name.match(/layers\.(\d+)\./);
      return m ? parseInt(m[1], 10) : -1;
    })
    .filter((i) => i >= 0);
  const numLayers =
    layerIndices.length > 0 ? Math.max(...layerIndices) + 1 : 30;

  // Detect head count from Q projection shape
  const qProj = tensors.find((t) => t.name.includes("q_proj.weight"));
  const numHeads = qProj
    ? qProj.shape[0] / (hiddenSize / 32)
    : 32;

  const kvProj = tensors.find((t) => t.name.includes("k_proj.weight"));
  const kvDim = kvProj?.shape[0] ?? hiddenSize;
  const headDim = hiddenSize / numHeads;
  const numKVHeads = kvDim / headDim;

  const isOfficial = vocabSize > 100000;

  return {
    modelType: "bitnet",
    vocabSize,
    hiddenSize,
    intermediateSize: 0, // Will be inferred from FFN weight shapes
    numHiddenLayers: numLayers,
    numAttentionHeads: numHeads,
    numKeyValueHeads: numKVHeads,
    maxPositionEmbeddings: 4096,
    rmsNormEps: 1e-5,
    ropeTheta: isOfficial ? 500000.0 : 10000.0,
    tieWordEmbeddings: false,
    activation: isOfficial ? "relu2" : "silu",
  };
}

/**
 * Fetch with Cache API support.
 * On first load, stores in Cache API for instant subsequent loads.
 */
const IDB_NAME = "0xbitnet";
const IDB_STORE = "models";

function openModelDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idbGet(db: IDBDatabase, key: string): Promise<ArrayBuffer | undefined> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, "readonly");
    const req = tx.objectStore(IDB_STORE).get(key);
    req.onsuccess = () => resolve(req.result as ArrayBuffer | undefined);
    req.onerror = () => reject(req.error);
  });
}

function idbPut(db: IDBDatabase, key: string, value: ArrayBuffer): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, "readwrite");
    tx.objectStore(IDB_STORE).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function fetchModel(
  url: string,
  onProgress: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
  // Try IndexedDB cache first
  if (typeof indexedDB !== "undefined") {
    try {
      const db = await openModelDB();
      const cached = await idbGet(db, url);
      db.close();
      if (cached) {
        onProgress(cached.byteLength, cached.byteLength);
        return cached;
      }
    } catch {
      // IndexedDB unavailable — fall through to fetch
    }
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
  }

  const contentLength = parseInt(
    response.headers.get("content-length") ?? "0",
    10
  );
  const reader = response.body?.getReader();
  if (!reader) {
    const buffer = await response.arrayBuffer();
    onProgress(buffer.byteLength, buffer.byteLength);
    return buffer;
  }

  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, contentLength);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }

  // Save to IndexedDB for next time
  if (typeof indexedDB !== "undefined") {
    try {
      const db = await openModelDB();
      await idbPut(db, url, buffer.buffer);
      db.close();
    } catch {
      // Quota exceeded — skip
    }
  }

  return buffer.buffer;
}
