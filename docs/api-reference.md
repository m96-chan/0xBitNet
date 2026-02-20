# API Reference

## `BitNet` (class)

The high-level entry point for loading and running BitNet models.

### `BitNet.load(source, options?)`

Load a BitNet model from a URL.

```typescript
static async load(source: string | URL, options?: LoadOptions): Promise<BitNet>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `string \| URL` | URL to a GGUF or Safetensors file |
| `options` | `LoadOptions` | Loading options (see below) |

**Returns:** `Promise<BitNet>`

### `bitnet.generate(prompt, options?)`

Generate text from a prompt. Yields tokens as they are generated.

```typescript
async *generate(
  prompt: string | ChatMessage[],
  options?: GenerateOptions
): AsyncGenerator<string>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `string \| ChatMessage[]` | Plain text or chat messages |
| `options` | `GenerateOptions` | Generation options (see below) |

**Returns:** `AsyncGenerator<string>`

### `bitnet.diagnose(prompt?)`

Run GPU diagnostics: a forward pass with per-stage tensor readback.

```typescript
async diagnose(prompt?: string): Promise<DiagnosticResult[]>
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `string` | `"Hello"` | Input text to run through the model |

**Returns:** `Promise<DiagnosticResult[]>`

### `bitnet.dispose()`

Release all GPU resources held by this instance. Must be called when the model is no longer needed.

```typescript
dispose(): void
```

---

## Standalone Functions

### `initGPU(existingDevice?)`

Initialize a WebGPU adapter and device with maximum limits for large model support.

```typescript
async function initGPU(existingDevice?: GPUDevice): Promise<GPUContext>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `existingDevice` | `GPUDevice` | Optional existing device to reuse |

**Returns:** `Promise<GPUContext>`

**Throws:** `GPUDeviceError` if WebGPU is unavailable or adapter/device creation fails.

### `listCachedModels()`

List all model URLs cached in IndexedDB.

```typescript
async function listCachedModels(): Promise<string[]>
```

**Returns:** `Promise<string[]>` — Array of cached model URLs.

### `deleteCachedModel(url)`

Delete a cached model from IndexedDB.

```typescript
async function deleteCachedModel(url: string): Promise<void>
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `string` | The model URL to remove from cache |

---

## Interfaces

### `LoadOptions`

```typescript
interface LoadOptions {
  device?: GPUDevice;
  format?: WeightFormat;        // "gguf" | "safetensors"
  onProgress?: (progress: LoadProgress) => void;
  signal?: AbortSignal;
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | `GPUDevice` | Auto-created | Existing GPU device to reuse |
| `format` | `WeightFormat` | Auto-detected | Force weight format |
| `onProgress` | `(progress: LoadProgress) => void` | — | Progress callback |
| `signal` | `AbortSignal` | — | Abort signal to cancel loading |

### `GenerateOptions`

```typescript
interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  repeatPenalty?: number;
  repeatLastN?: number;
  onToken?: (token: string) => void;
  signal?: AbortSignal;
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `maxTokens` | `number` | `256` | Maximum tokens to generate |
| `temperature` | `number` | `1.0` | Sampling temperature |
| `topK` | `number` | `50` | Top-K sampling (0 = disabled) |
| `repeatPenalty` | `number` | `1.0` | Repetition penalty (1.0 = disabled) |
| `repeatLastN` | `number` | `64` | Window size for repetition penalty |
| `onToken` | `(token: string) => void` | — | Callback fired for each token |
| `signal` | `AbortSignal` | — | Abort signal to cancel generation |

### `LoadProgress`

```typescript
interface LoadProgress {
  phase: "download" | "parse" | "upload";
  loaded: number;
  total: number;
  fraction: number;             // 0.0 – 1.0
}
```

### `ChatMessage`

```typescript
interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}
```

### `DiagnosticResult`

```typescript
interface DiagnosticResult {
  name: string;
  length: number;
  min: number;
  max: number;
  mean: number;
  rms: number;
  nanCount: number;
  infCount: number;
  zeroCount: number;
  first8: number[];
}
```

### `ModelConfig`

```typescript
interface ModelConfig {
  modelType: "bitnet";
  vocabSize: number;
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  maxPositionEmbeddings: number;
  rmsNormEps: number;
  ropeTheta: number;
  tieWordEmbeddings: boolean;
  activation: "relu2" | "silu" | "swiglu";
}
```

### `GPUContext`

```typescript
interface GPUContext {
  device: GPUDevice;
  adapter: GPUAdapter | null;
  limits: GPUSupportedLimits;
}
```

---

## Predefined Configs

Import these configs to inspect model parameters or pass to lower-level APIs.

### `BITNET_2B_4T_CONFIG`

Config for [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T).

```typescript
{
  modelType: "bitnet",
  vocabSize: 128256,
  hiddenSize: 2560,
  intermediateSize: 6912,
  numHiddenLayers: 30,
  numAttentionHeads: 20,
  numKeyValueHeads: 5,
  maxPositionEmbeddings: 4096,
  rmsNormEps: 1e-5,
  ropeTheta: 500000.0,
  tieWordEmbeddings: true,
  activation: "relu2",
}
```

### `BITNET_0_7B_CONFIG`

Config for [1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large) (0.7B).

```typescript
{
  modelType: "bitnet",
  vocabSize: 32002,
  hiddenSize: 2048,
  intermediateSize: 5632,
  numHiddenLayers: 24,
  numAttentionHeads: 32,
  numKeyValueHeads: 32,
  maxPositionEmbeddings: 2048,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  tieWordEmbeddings: false,
  activation: "silu",
}
```

### `BITNET_3B_CONFIG`

Config for [HF1BitLLM/bitnet_b1_58-3B](https://huggingface.co/HF1BitLLM/bitnet_b1_58-3B).

```typescript
{
  modelType: "bitnet",
  vocabSize: 32002,
  hiddenSize: 3200,
  intermediateSize: 8640,
  numHiddenLayers: 26,
  numAttentionHeads: 32,
  numKeyValueHeads: 32,
  maxPositionEmbeddings: 2048,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  tieWordEmbeddings: false,
  activation: "silu",
}
```

---

## Error Classes

### `GPUDeviceError`

Thrown by `initGPU()` when WebGPU is unavailable or adapter/device creation fails.

```typescript
class GPUDeviceError extends Error {
  name: "GPUDeviceError";
}
```

---

## Re-exports

The following internal types are also exported for advanced usage:

| Export | Kind | Description |
|--------|------|-------------|
| `WeightStore` | class | GPU buffer storage for model weights |
| `loadModel()` | function | Low-level model loader (GGUF/Safetensors) |
| `LoadResult` | type | Return type of `loadModel()` |
| `BitNetModel` | class | Low-level transformer model |
| `Tokenizer` | class | BPE tokenizer with chat template support |
| `WeightFormat` | type | `"gguf" \| "safetensors"` |
| `TokenizerConfig` | type | Tokenizer configuration |
| `WorkerRequest` | type | Web Worker message types |
| `WorkerResponse` | type | Web Worker response types |
