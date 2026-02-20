# oxbitnet-swift

Swift bindings for [oxbitnet](https://crates.io/crates/oxbitnet) — run [BitNet b1.58](https://github.com/microsoft/BitNet) ternary LLMs with GPU acceleration (wgpu).

Part of [0xBitNet](https://github.com/m96-chan/0xBitNet).

## Build

First, build the native library:

```bash
cargo build -p oxbitnet-ffi --release
```

Produces `target/release/liboxbitnet_ffi.so` (Linux) / `.dylib` (macOS) / `oxbitnet_ffi.dll` (Windows).

Then build the Swift package:

```bash
cd packages/rust/crates/oxbitnet-swift
swift build -Xlinker -L../../../../target/release
```

## Quick Start

```swift
import OxBitNet

let model = try await BitNet.load("model.gguf")

// Raw prompt
for try await token in model.generate("Hello!") {
    print(token, terminator: "")
}

// Chat messages
for try await token in model.chat([.user("Hello!")], options: .init(temperature: 0.7)) {
    print(token, terminator: "")
}

model.dispose()
```

## API

### Loading

```swift
// Async (recommended)
let model = try await BitNet.load("model.gguf")

// With progress
let model = try await BitNet.load("model.gguf", options: LoadOptions(
    onProgress: { p in
        print("[\(p.phase)] \(String(format: "%.1f", p.fraction * 100))%")
    }
))

// Sync (blocks calling thread)
let model = try BitNet.loadSync("model.gguf")
```

### Generation

```swift
// Raw prompt — returns AsyncThrowingStream<String, Error>
for try await token in model.generate("Once upon a time") {
    print(token, terminator: "")
}

// With options
for try await token in model.generate("Hello!", options: GenerateOptions(
    maxTokens: 512,
    temperature: 0.7,
    topK: 40
)) {
    print(token, terminator: "")
}
```

### Chat

```swift
let messages: [ChatMessage] = [
    .system("You are a helpful assistant."),
    .user("What is 2+2?"),
]

for try await token in model.chat(messages) {
    print(token, terminator: "")
}
```

### Cleanup

`BitNet` calls `dispose()` automatically in its `deinit`, but you can call it explicitly:

```swift
model.dispose()
```

### Logging

```swift
setLogger(minLevel: .info) { level, message in
    print("[\(level)] \(message)")
}
```

Must be called before `BitNet.load`. Can only be called once.

## Generation Options

| Field | Default | Description |
|-------|---------|-------------|
| `maxTokens` | 256 | Maximum tokens to generate |
| `temperature` | 1.0 | Sampling temperature |
| `topK` | 50 | Top-k sampling |
| `repeatPenalty` | 1.1 | Repetition penalty |
| `repeatLastN` | 64 | Window for repetition penalty |

## Running the Example

```bash
cd packages/rust
cargo build -p oxbitnet-ffi --release
cd crates/oxbitnet-swift
swift run -Xlinker -L../../../../target/release Chat model.gguf "Hello!"
```

## License

MIT
