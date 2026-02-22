<p align="center">
  <img src="https://raw.githubusercontent.com/m96-chan/0xBitNet/main/assets/hero.png" alt="0xBitNet — 1-bit Inference on WebGPU" width="720" />
</p>

<h1 align="center">0xBitNet</h1>

<p align="center">
  <a href="https://github.com/m96-chan/0xBitNet/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/m96-chan/0xBitNet/ci.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://github.com/m96-chan/0xBitNet/actions/workflows/rust-ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/m96-chan/0xBitNet/rust-ci.yml?branch=main&label=Rust%20CI" alt="Rust CI"></a>
  <a href="https://www.npmjs.com/package/0xbitnet"><img src="https://img.shields.io/npm/v/0xbitnet" alt="npm"></a>
  <a href="https://crates.io/crates/oxbitnet"><img src="https://img.shields.io/crates/v/oxbitnet" alt="crates.io"></a>
  <a href="https://pypi.org/project/oxbitnet/"><img src="https://img.shields.io/pypi/v/oxbitnet" alt="PyPI"></a>
  <a href="https://central.sonatype.com/artifact/io.github.m96-chan/OxBitNet"><img src="https://img.shields.io/maven-central/v/io.github.m96-chan/OxBitNet" alt="Maven Central"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/m96-chan/0xBitNet" alt="License"></a>
</p>

<p align="center">
  <strong>Run <a href="https://github.com/microsoft/BitNet">Microsoft BitNet b1.58</a> ternary LLMs with WebGPU — in browsers and native apps.</strong>
</p>

<p align="center">
  <a href="https://m96-chan.github.io/0xBitNet/chat/">Live Chat Demo</a> · <a href="https://m96-chan.github.io/0xBitNet/tldr/">TL;DR Widget Demo</a> · <a href="docs/getting-started.md">Getting Started</a> · <a href="docs/api-reference.md">API Reference</a>
</p>

---

0xBitNet runs BitNet b1.58 ternary LLMs on WebGPU. Custom WGSL compute kernels handle the ternary matrix operations, with bindings for TypeScript, Rust, and Python. Works in browsers, Node.js, and native apps.

## Highlights

- **Pure WebGPU** — Custom WGSL kernels for ternary matrix operations (no WASM, no server)
- **Multi-language** — TypeScript (`0xbitnet`), Rust (`oxbitnet`), Python (`oxbitnet`), Swift (`OxBitNet`), Java/Android (`oxbitnet-java`), C (`oxbitnet-ffi`)
- **Cross-platform** — Browsers, Node.js, Deno, native apps via wgpu
- **Chat templates** — Built-in LLaMA 3 chat message formatting
- **Automatic caching** — IndexedDB (browser) / disk cache (native)
- **Streaming** — Token-by-token output via async generators / streams / callbacks

## Quick Start

### TypeScript / JavaScript

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

### Rust

```bash
cargo add oxbitnet
```

```rust
use oxbitnet::BitNet;
use futures::StreamExt;

let mut model = BitNet::load("model.gguf", Default::default()).await?;

let mut stream = model.generate("Hello!", Default::default());
while let Some(token) = stream.next().await {
    print!("{token}");
}

model.dispose();
```

### Python

```bash
pip install oxbitnet
```

```python
from oxbitnet import BitNet

model = BitNet.load_sync("model.gguf")

model.chat(
    [("system", "You are a helpful assistant."), ("user", "Hello!")],
    on_token=lambda t: print(t, end="", flush=True),
    temperature=0.7,
)

model.dispose()
```

### Swift

```swift
import OxBitNet

let model = try await BitNet.load("model.gguf")

for try await token in model.chat([.user("Hello!")], options: .init(temperature: 0.7)) {
    print(token, terminator: "")
}

model.dispose()
```

### Java

```java
import io.github.m96chan.oxbitnet.*;
import java.util.List;

try (BitNet model = BitNet.loadSync("model.gguf")) {
    model.chat(
        List.of(new ChatMessage("user", "Hello!")),
        token -> {
            System.out.print(token);
            return true;
        },
        new GenerateOptions().temperature(0.7f)
    );
}
```

### C / FFI

```c
#include "oxbitnet.h"

static int32_t on_token(const char *token, uintptr_t len, void *userdata) {
    fwrite(token, 1, len, stdout);
    return 0; /* 0 = continue, non-zero = stop */
}

int main(void) {
    OxBitNet *model = oxbitnet_load("model.gguf", NULL);

    OxBitNetChatMessage messages[] = {
        { .role = "user", .content = "Hello!" },
    };
    OxBitNetGenerateOptions opts = oxbitnet_default_generate_options();

    oxbitnet_chat(model, messages, 1, &opts, on_token, NULL);
    oxbitnet_free(model);
}
```

### Chat Messages (TypeScript)

```typescript
const messages = [
  { role: "system" as const, content: "You are a helpful assistant." },
  { role: "user" as const, content: "Explain quantum computing in one sentence." },
];

for await (const token of model.generate(messages, { maxTokens: 128, temperature: 0.7 })) {
  process.stdout.write(token);
}
```

## Supported Models

| Model | GGUF | Parameters | VRAM |
|-------|------|------------|------|
| [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | [ggml-model-i2_s.gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) | 2B | ~1.5 GB |
| [Falcon-E 1B Instruct](https://huggingface.co/tiiuae/Falcon-E-1B-Instruct-GGUF) | [ggml-model-i2_s.gguf](https://huggingface.co/tiiuae/Falcon-E-1B-Instruct-GGUF) | 1B | ~666 MB |
| [Falcon-E 3B Instruct](https://huggingface.co/tiiuae/Falcon-E-3B-Instruct-GGUF) | [ggml-model-i2_s.gguf](https://huggingface.co/tiiuae/Falcon-E-3B-Instruct-GGUF) | 3B | ~1 GB |

Any I2_S GGUF model with a compatible architecture should work — see [Model Compatibility](docs/model-compatibility.md) for details.

## Install

| Language | Package | Install |
|----------|---------|---------|
| TypeScript / JS | [`0xbitnet`](https://www.npmjs.com/package/0xbitnet) | `npm install 0xbitnet` |
| Rust | [`oxbitnet`](https://crates.io/crates/oxbitnet) | `cargo add oxbitnet` |
| Python | [`oxbitnet`](https://pypi.org/project/oxbitnet/) | `pip install oxbitnet` |
| Swift / iOS | `OxBitNet` | Swift Package Manager (see [oxbitnet-swift](packages/rust/crates/oxbitnet-swift/)) |
| Java / Android | `oxbitnet-java` | `cargo build -p oxbitnet-java --release` |
| C / FFI | `oxbitnet-ffi` | `cargo build -p oxbitnet-ffi --release` |

## API Overview

### TypeScript

| Method | Description |
|--------|-------------|
| `BitNet.load(url, options?)` | Load a GGUF model from a URL |
| `bitnet.generate(prompt, options?)` | Stream tokens as an `AsyncGenerator<string>` |
| `bitnet.diagnose(prompt?)` | Run GPU diagnostics on a forward pass |
| `bitnet.dispose()` | Release all GPU resources |

### Rust

| Method | Description |
|--------|-------------|
| `BitNet::load(source, options).await` | Load a GGUF model |
| `bitnet.generate(prompt, options)` | Stream tokens as `impl Stream<Item = String>` |
| `bitnet.generate_chat(messages, options)` | Chat with template formatting |
| `bitnet.dispose()` | Release all GPU resources |

### Python

| Method | Description |
|--------|-------------|
| `BitNet.load_sync(source)` | Load a GGUF model |
| `model.chat(messages, on_token)` | Chat with streaming callback |
| `model.generate(prompt, on_token)` | Generate with streaming callback |
| `model.generate_sync(prompt)` | Generate, return full string |
| `model.dispose()` | Release all GPU resources |

### Swift

| Method | Description |
|--------|-------------|
| `BitNet.load(source, options:)` | Load a GGUF model (async) |
| `BitNet.loadSync(source, options:)` | Load a GGUF model (blocking) |
| `model.generate(prompt, options:)` | Stream tokens as `AsyncThrowingStream<String, Error>` |
| `model.chat(messages, options:)` | Chat with streaming via `AsyncThrowingStream` |
| `model.dispose()` | Release all GPU resources (also called by `deinit`) |

### Java

| Method | Description |
|--------|-------------|
| `BitNet.loadSync(source, options?)` | Load a GGUF model |
| `model.chat(messages, callback, options?)` | Chat with streaming callback |
| `model.generate(prompt, callback, options?)` | Generate with streaming callback |
| `model.dispose()` / `model.close()` | Release all GPU resources (AutoCloseable) |

### C / FFI

| Function | Description |
|----------|-------------|
| `oxbitnet_load(source, options)` | Load a GGUF model, returns opaque handle |
| `oxbitnet_chat(model, messages, n, opts, cb, ud)` | Chat with streaming callback |
| `oxbitnet_generate(model, prompt, opts, cb, ud)` | Generate with streaming callback |
| `oxbitnet_free(model)` | Release all GPU resources |
| `oxbitnet_error_message()` | Get last error (thread-local) |

## Platform Support

0xBitNet runs on any platform with a [WebGPU](https://www.w3.org/TR/webgpu/) implementation:

**Browsers:**
- Chrome / Edge 113+ (recommended)
- Firefox Nightly (behind flag)
- Safari 18+

**Native (Rust / Python):**
- Uses [wgpu](https://wgpu.rs/) — Vulkan, Metal, DX12 backends automatically
- No browser or WebGPU runtime needed

**Native (Node.js / Deno):**
- Deno (built-in WebGPU)
- Node.js with [`webgpu`](https://www.npmjs.com/package/webgpu) npm package (Dawn bindings) — see [Node.js CLI example](examples/node-cli/)
- Any runtime exposing the WebGPU API (e.g., wgpu-native, Electron)

A dedicated GPU with sufficient VRAM is required (see [Supported Models](#supported-models) for estimates).

## Examples

### [Web Chat](https://m96-chan.github.io/0xBitNet/chat/)

A WebGPU-powered chat application. Downloads the model on first visit, then runs LLM chat completely on-device — no backend needed.

### [TL;DR Widget](https://m96-chan.github.io/0xBitNet/tldr/)

An offline-ready summarization widget. Provides LLM-powered TL;DR without any network dependency.

### [Node.js CLI](examples/node-cli/)

Run BitNet from the command line using Node.js and the [`webgpu`](https://www.npmjs.com/package/webgpu) npm package (Dawn bindings). Interactive chat with streaming output and tok/s metrics.

```bash
cd examples/node-cli
npm install && npm start
```

### Rust CLI

Interactive chat using native wgpu.

```bash
cd packages/rust
cargo run --example chat --release
```

### Python CLI

Interactive chat via Python bindings.

```bash
pip install oxbitnet
python packages/rust/crates/oxbitnet-python/examples/chat.py
```

### Swift CLI

Minimal Swift chat example wrapping the C FFI layer.

```bash
cd packages/rust
cargo build -p oxbitnet-ffi --release
cd crates/oxbitnet-swift
swift run -Xlinker -L../../../../target/release Chat model.gguf "Hello!"
```

### Java CLI

Minimal Java chat example using JNI bindings.

```bash
cd packages/rust
cargo build -p oxbitnet-java --release
cd crates/oxbitnet-java/examples
javac -cp ../java/src/main/java:. Chat.java
java -Djava.library.path=../../../../target/release -cp ../java/src/main/java:. Chat model.gguf "Hello!"
```

### C CLI

Minimal C example using the FFI bindings.

```bash
cd packages/rust
cargo build -p oxbitnet-ffi --release
gcc crates/oxbitnet-ffi/examples/chat.c -Icrates/oxbitnet-ffi -Ltarget/release -loxbitnet_ffi -o chat
LD_LIBRARY_PATH=target/release ./chat model.gguf "Hello!"
```

## Architecture

```
0xbitnet/
├── packages/
│   ├── core/               # WGSL kernels + TypeScript API (npm: 0xbitnet)
│   │   └── src/
│   │       ├── gpu/        # WebGPU device init, buffer pool
│   │       ├── model/      # GGUF parser, weight loader, config
│   │       ├── nn/         # Transformer layers, attention, BitLinear
│   │       ├── shaders/    # 12 WGSL compute shaders (shared with Rust)
│   │       └── tokenizer/  # BPE tokenizer, chat templates
│   └── rust/               # Rust + Python bindings
│       └── crates/
│           ├── oxbitnet/           # Rust library (crates.io: oxbitnet)
│           ├── oxbitnet-python/    # Python bindings via PyO3 (PyPI: oxbitnet)
│           ├── oxbitnet-swift/     # Swift bindings via C FFI (SPM package)
│           ├── oxbitnet-java/      # Java/JNI bindings (Android-ready)
│           └── oxbitnet-ffi/       # C FFI bindings (cdylib + staticlib)
├── examples/
│   ├── web-chat/           # Chat app demo (Vite)
│   ├── tl-dr-widget/       # Offline TL;DR widget demo (Vite)
│   └── node-cli/           # Node.js CLI using Dawn WebGPU bindings
└── docs/
```

See [Architecture](docs/architecture.md) for data flow and internals.

## Prerequisites

- **TypeScript/JS**: Node.js 18+, a WebGPU-capable environment
- **Rust**: Rust 1.75+, a Vulkan/Metal/DX12-capable GPU
- **Swift**: Swift 5.9+, a Vulkan/Metal/DX12-capable GPU
- **Java**: JDK 17+, a Vulkan/Metal/DX12-capable GPU
- **Python**: Python 3.9+, `pip install oxbitnet`

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
