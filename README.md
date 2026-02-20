<p align="center">
  <img src="assets/hero.png" alt="0xBitNet — 1-bit Inference on WebGPU" width="720" />
</p>

<h1 align="center">0xBitNet</h1>

<p align="center">
  <a href="https://www.npmjs.com/package/0xbitnet"><img src="https://img.shields.io/npm/v/0xbitnet" alt="npm"></a>
  <a href="https://github.com/m96-chan/0xBitNet/actions"><img src="https://img.shields.io/github/actions/workflow/status/m96-chan/0xBitNet/ci.yml?branch=main" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/m96-chan/0xBitNet" alt="License"></a>
</p>

<p align="center">
  <strong>Run <a href="https://github.com/microsoft/BitNet">Microsoft BitNet b1.58</a> ternary LLMs with WebGPU — in browsers and native apps.</strong>
</p>

<p align="center">
  <a href="https://m96-chan.github.io/0xBitNet/chat/">Live Chat Demo</a> · <a href="https://m96-chan.github.io/0xBitNet/tldr/">TL;DR Widget Demo</a> · <a href="docs/getting-started.md">Getting Started</a> · <a href="docs/api-reference.md">API Reference</a>
</p>

---

0xBitNet is a TypeScript library for 1-bit LLM inference on WebGPU. It implements BitNet's ternary compute kernels in WGSL (WebGPU Shading Language) and wraps them in an ergonomic TypeScript API. Works in any environment with a WebGPU device — browsers, Deno, Node.js with WebGPU-native bindings, or embedded via wgpu/Dawn.

## Highlights

- **Pure WebGPU** — Custom WGSL kernels for ternary matrix operations (no WASM, no server)
- **Cross-platform** — Runs anywhere WebGPU is available: browsers, Deno, Node.js, native apps
- **TypeScript-first** — Type-safe API with full ESM and CJS support
- **Chat templates** — Built-in chat message formatting with `ChatMessage[]`
- **Automatic caching** — Models are cached in IndexedDB (browser) after first download
- **Offline-capable** — Works without a network connection after the initial model download
- **NPM package** — `npm install 0xbitnet`

## Quick Start

```bash
npm install 0xbitnet
```

```typescript
import { BitNet } from "0xbitnet";

const model = await BitNet.load(
  "https://huggingface.co/m96-chan/bitnet-b1.58-2B-4T-gguf/resolve/main/bitnet-b1.58-2B-4T.gguf",
  { onProgress: (p) => console.log(`${p.phase}: ${(p.fraction * 100).toFixed(1)}%`) }
);

for await (const token of model.generate("The meaning of life is")) {
  process.stdout.write(token);
}

model.dispose();
```

### Chat Messages

```typescript
const messages = [
  { role: "system" as const, content: "You are a helpful assistant." },
  { role: "user" as const, content: "Explain quantum computing in one sentence." },
];

for await (const token of model.generate(messages, { maxTokens: 128, temperature: 0.7 })) {
  process.stdout.write(token);
}
```

### Cache Management

In browsers, models are automatically cached in IndexedDB after the first download.

```typescript
import { listCachedModels, deleteCachedModel } from "0xbitnet";

const cached = await listCachedModels();
console.log("Cached models:", cached);

await deleteCachedModel("https://example.com/model.gguf");
```

## Supported Models

| Model | Config | Parameters | VRAM |
|-------|--------|------------|------|
| [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | `BITNET_2B_4T_CONFIG` | 2B | ~1.5 GB |
| [1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large) | `BITNET_0_7B_CONFIG` | 0.7B | ~0.5 GB |
| [HF1BitLLM/bitnet_b1_58-3B](https://huggingface.co/HF1BitLLM/bitnet_b1_58-3B) | `BITNET_3B_CONFIG` | 3B | ~2 GB |

Third-party models using the I2_S ternary format (e.g., Falcon-Edge 1B/3B, Aramis-2B) are auto-detected from GGUF metadata. See [Model Compatibility](docs/model-compatibility.md) for details.

## API Overview

The main entry point is the `BitNet` class:

| Method | Description |
|--------|-------------|
| `BitNet.load(url, options?)` | Load a GGUF model from a URL |
| `bitnet.generate(prompt, options?)` | Stream tokens as an `AsyncGenerator<string>` |
| `bitnet.diagnose(prompt?)` | Run GPU diagnostics on a forward pass |
| `bitnet.dispose()` | Release all GPU resources |

Standalone functions: `initGPU()`, `listCachedModels()`, `deleteCachedModel(url)`

Full details in the [API Reference](docs/api-reference.md).

## Platform Support

0xBitNet runs on any platform with a [WebGPU](https://www.w3.org/TR/webgpu/) implementation:

**Browsers:**
- Chrome / Edge 113+ (recommended)
- Firefox Nightly (behind flag)
- Safari 18+

**Native:**
- Deno (built-in WebGPU)
- Node.js with [wgpu](https://github.com/gfx-rs/wgpu) or [Dawn](https://dawn.googlesource.com/dawn) bindings
- Any runtime exposing the WebGPU API (e.g., wgpu-native, Electron)

A dedicated GPU with sufficient VRAM is required (see [Supported Models](#supported-models) for estimates).

## Examples

### [Web Chat](https://m96-chan.github.io/0xBitNet/chat/)

A WebGPU-powered chat application. Downloads the model on first visit, then runs LLM chat completely on-device — no backend needed.

### [TL;DR Widget](https://m96-chan.github.io/0xBitNet/tldr/)

An offline-ready summarization widget. Provides LLM-powered TL;DR without any network dependency.

## Architecture

```
0xbitnet/
├── packages/core/          # WGSL kernels + TypeScript API (npm: 0xbitnet)
│   └── src/
│       ├── gpu/            # WebGPU device init, buffer pool
│       ├── model/          # GGUF/Safetensors parser, weight loader, config
│       ├── nn/             # Transformer layers, attention, BitLinear
│       ├── shaders/        # WGSL compute shaders
│       ├── tokenizer/      # BPE tokenizer, chat templates
│       └── worker/         # Worker thread support
├── examples/
│   ├── web-chat/           # Chat app demo (Vite)
│   └── tl-dr-widget/       # Offline TL;DR widget demo (Vite)
└── docs/                   # Documentation
    ├── getting-started.md
    ├── api-reference.md
    ├── architecture.md
    └── model-compatibility.md
```

See [Architecture](docs/architecture.md) for data flow and internals.

## Prerequisites

- Node.js 18+
- A WebGPU-capable environment (see [Platform Support](#platform-support))

## Contributing

Contributions are welcome! Whether it's a bug report, feature request, or pull request — all input is appreciated.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

[MIT](LICENSE)
