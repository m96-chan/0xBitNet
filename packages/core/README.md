<p align="center">
  <img src="https://raw.githubusercontent.com/m96-chan/0xBitNet/main/assets/hero.png" alt="0xBitNet — 1-bit Inference on WebGPU" width="720" />
</p>

<h1 align="center">0xBitNet</h1>

<p align="center">
  <a href="https://www.npmjs.com/package/0xbitnet"><img src="https://img.shields.io/npm/v/0xbitnet" alt="npm"></a>
  <a href="https://crates.io/crates/oxbitnet"><img src="https://img.shields.io/crates/v/oxbitnet" alt="crates.io"></a>
  <a href="https://pypi.org/project/oxbitnet/"><img src="https://img.shields.io/pypi/v/oxbitnet" alt="PyPI"></a>
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

0xBitNet runs BitNet b1.58 ternary LLMs on WebGPU. Custom WGSL compute kernels handle the ternary matrix operations, with bindings for TypeScript, Rust, and Python.

Also available as: [`oxbitnet`](https://crates.io/crates/oxbitnet) (Rust) / [`oxbitnet`](https://pypi.org/project/oxbitnet/) (Python)

## Highlights

- **Pure WebGPU** — Custom WGSL kernels for ternary matrix operations (no WASM, no server)
- **Multi-language** — TypeScript (`0xbitnet`), Rust (`oxbitnet`), Python (`oxbitnet`)
- **Cross-platform** — Browsers, Node.js, Deno, native apps via wgpu
- **Chat templates** — Built-in LLaMA 3 chat message formatting
- **Automatic caching** — IndexedDB (browser) / disk cache (native)
- **Streaming** — Token-by-token output via async generators / streams / callbacks

## Quick Start

```bash
npm install 0xbitnet
```

```typescript
import { BitNet } from "0xbitnet";

const model = await BitNet.load(
  "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf",
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

| Model | GGUF | Parameters | VRAM |
|-------|------|------------|------|
| [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | [ggml-model-i2_s.gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) | 2B | ~1.5 GB |

More models are planned — see [#1](https://github.com/m96-chan/0xBitNet/issues/1) and [Model Compatibility](docs/model-compatibility.md) for GGUF requirements.

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
