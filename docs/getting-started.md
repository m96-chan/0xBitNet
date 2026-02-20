# Getting Started

## Requirements

- **Browser:** Chrome 113+, Edge 113+, Firefox Nightly, or Safari 18+ with [WebGPU](https://caniuse.com/webgpu)
- **Node.js:** 18+ (for build tooling / development)
- **GPU VRAM:** Depends on model size — see estimates below

| Model | Parameters | Approximate VRAM |
|-------|------------|-----------------|
| bitnet_b1_58-large (0.7B) | 0.7B | ~0.5 GB |
| bitnet-b1.58-2B-4T | 2B | ~1.5 GB |
| bitnet_b1_58-3B | 3B | ~2 GB |

## Installation

```bash
npm install 0xbitnet
```

## Load a Model

Pass a URL to a GGUF file. The model is downloaded, parsed, and uploaded to the GPU automatically. On subsequent loads the cached copy in IndexedDB is used — no re-download needed.

```typescript
import { BitNet } from "0xbitnet";

const model = await BitNet.load(
  "https://huggingface.co/m96-chan/bitnet-b1.58-2B-4T-gguf/resolve/main/bitnet-b1.58-2B-4T.gguf"
);
```

### Progress Tracking

```typescript
const model = await BitNet.load(url, {
  onProgress(p) {
    // p.phase: "download" | "parse" | "upload"
    // p.fraction: 0.0 – 1.0
    console.log(`${p.phase}: ${(p.fraction * 100).toFixed(1)}%`);
  },
});
```

The `LoadProgress` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `phase` | `"download" \| "parse" \| "upload"` | Current loading phase |
| `loaded` | `number` | Bytes/tensors processed so far |
| `total` | `number` | Total bytes/tensors |
| `fraction` | `number` | Progress ratio (0.0 – 1.0) |

## Generate Text (Streaming)

`generate()` returns an `AsyncGenerator<string>`, yielding one token at a time:

```typescript
for await (const token of model.generate("The meaning of life is")) {
  process.stdout.write(token);
}
```

### Generation Options

```typescript
for await (const token of model.generate("Once upon a time", {
  maxTokens: 512,       // default: 256
  temperature: 0.8,     // default: 1.0
  topK: 40,             // default: 50
  repeatPenalty: 1.1,   // default: 1.0
  repeatLastN: 64,      // default: 64
})) {
  process.stdout.write(token);
}
```

### Token Callback

If you prefer a callback style over `for await`, use `onToken`:

```typescript
const tokens: string[] = [];
// eslint-disable-next-line @typescript-eslint/no-unused-vars
for await (const _ of model.generate("Hello", {
  onToken(token) {
    tokens.push(token);
  },
})) {
  // tokens are also available via the callback
}
```

## Chat Messages

Pass an array of `ChatMessage` objects to use the built-in chat template:

```typescript
import type { ChatMessage } from "0xbitnet";

const messages: ChatMessage[] = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Explain quantum computing in one sentence." },
];

for await (const token of model.generate(messages, { maxTokens: 128 })) {
  process.stdout.write(token);
}
```

The `ChatMessage` interface:

```typescript
interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}
```

## Model Caching

Models are automatically cached in IndexedDB after the first download. Subsequent calls to `BitNet.load()` with the same URL will use the cached copy instantly.

### List Cached Models

```typescript
import { listCachedModels } from "0xbitnet";

const urls = await listCachedModels();
console.log("Cached:", urls);
// ["https://huggingface.co/.../model.gguf"]
```

### Delete a Cached Model

```typescript
import { deleteCachedModel } from "0xbitnet";

await deleteCachedModel("https://huggingface.co/.../model.gguf");
```

## Cancellation with AbortSignal

Both `load()` and `generate()` accept an `AbortSignal` for cancellation:

### Cancel Loading

```typescript
const controller = new AbortController();

// Cancel after 30 seconds
setTimeout(() => controller.abort(), 30_000);

try {
  const model = await BitNet.load(url, { signal: controller.signal });
} catch (err) {
  if (err instanceof DOMException && err.name === "AbortError") {
    console.log("Loading cancelled");
  }
}
```

### Cancel Generation

```typescript
const controller = new AbortController();

// Cancel after 100 tokens
let count = 0;
for await (const token of model.generate("Hello", { signal: controller.signal })) {
  process.stdout.write(token);
  if (++count >= 100) controller.abort();
}
```

## Cleanup

Always call `dispose()` when you're done with a model to release GPU resources:

```typescript
model.dispose();
```

## Next Steps

- [API Reference](api-reference.md) — Full API documentation
- [Architecture](architecture.md) — How 0xBitNet works internally
- [Model Compatibility](model-compatibility.md) — Supported models and GGUF requirements
