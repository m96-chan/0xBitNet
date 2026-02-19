<p align="center">
  <img src="assets/hero.png" alt="0xBitNet — 1-bit Inference on WebGPU" width="720" />
</p>

<h1 align="center">0xBitNet</h1>

<p align="center">
  <strong>Run <a href="https://github.com/microsoft/BitNet">Microsoft BitNet</a> inference entirely in the browser with WebGPU.</strong>
</p>

0xBitNet is a TypeScript library that brings 1-bit LLM inference to the web. It implements BitNet's compute kernels in WGSL (WebGPU Shading Language) and wraps them in an ergonomic TypeScript API, distributed as an NPM package.

## Highlights

- **Pure WebGPU** — Custom WGSL kernels for 1-bit quantized matrix operations
- **Fully client-side** — No server required; inference runs entirely in the browser
- **TypeScript-first** — Type-safe API designed for the web ecosystem
- **NPM package** — Install with `npm install 0xbitnet`
- **Offline-capable** — Works without a network connection after the initial model download

## Examples

### Web Chat

A browser-based chat application. Downloads the model on first visit, then runs LLM chat completely on the client — no backend needed.

### TL;DR Widget

An offline-ready summarization widget. Provides LLM-powered TL;DR without any network dependency.

## Getting Started

### Prerequisites

- A browser with [WebGPU support](https://caniuse.com/webgpu) (Chrome 113+, Edge 113+, Firefox Nightly)
- Node.js 18+

### Installation

```bash
npm install 0xbitnet
```

### Basic Usage

```typescript
import { BitNet } from "0xbitnet";

const model = await BitNet.load("/path/to/model");
const output = await model.generate("Hello, world!");
console.log(output);
```

> **Note:** The API is under active development and subject to change.

## Model Conversion

Use the official [BitNet repository](https://github.com/microsoft/BitNet) tools to convert models into the format 0xBitNet expects.

## Architecture

```
0xbitnet/
├── packages/core/       # WGSL kernels + TypeScript API (NPM package)
├── examples/
│   ├── web-chat/        # Sample 1 — Browser chat app
│   └── tl-dr-widget/    # Sample 2 — Offline TL;DR widget
└── docs/                # Documentation
```

## Contributing

Contributions are welcome! Whether it's a bug report, feature request, or pull request — all input is appreciated.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Roadmap

- [ ] Core WGSL kernels for BitNet inference
- [ ] TypeScript API wrapper
- [ ] NPM package publishing
- [ ] Web Chat example
- [ ] TL;DR Widget example
- [ ] Performance benchmarks
- [ ] Support for additional BitNet model variants

## License

[MIT](LICENSE)
