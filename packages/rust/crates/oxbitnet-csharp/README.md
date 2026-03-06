# oxbitnet-csharp

C# / .NET bindings for [oxbitnet](https://crates.io/crates/oxbitnet) — run [BitNet b1.58](https://github.com/microsoft/BitNet) ternary LLMs with GPU acceleration (wgpu).

Part of [0xBitNet](https://github.com/m96-chan/0xBitNet).

## Build

First, build the native library:

```bash
cargo build -p oxbitnet-ffi --release
```

Produces `target/release/liboxbitnet_ffi.so` (Linux) / `.dylib` (macOS) / `oxbitnet_ffi.dll` (Windows).

Then build the C# project:

```bash
cd packages/rust/crates/oxbitnet-csharp
dotnet build
```

## Quick Start

```csharp
using OxBitNet;

// Load a model
using var model = BitNet.LoadSync("model.gguf");

// Raw prompt
model.Generate("Hello!", token => Console.Write(token));

// Chat messages
model.Chat(new[] {
    ChatMessage.User("Hello!")
}, token => Console.Write(token), new GenerateOptions { Temperature = 0.7f });
```

## API

### Loading

```csharp
// Sync (blocks calling thread)
using var model = BitNet.LoadSync("model.gguf");

// Async
using var model = await BitNet.Load("model.gguf");

// With progress
using var model = BitNet.LoadSync("model.gguf", new LoadOptions {
    OnProgress = p => Console.WriteLine($"[{p.Phase}] {p.Fraction * 100:F1}%")
});
```

### Generation

```csharp
// Raw prompt — tokens delivered via callback
model.Generate("Once upon a time", token => Console.Write(token));

// With options
model.Generate("Hello!", token => Console.Write(token), new GenerateOptions {
    MaxTokens = 512,
    Temperature = 0.7f,
    TopK = 40,
});

// Async variant
await model.GenerateAsync("Hello!", token => Console.Write(token));
```

### Chat

```csharp
var messages = new[] {
    ChatMessage.System("You are a helpful assistant."),
    ChatMessage.User("What is 2+2?"),
};

model.Chat(messages, token => Console.Write(token));

// Async variant
await model.ChatAsync(messages, token => Console.Write(token));
```

### Cleanup

`BitNet` implements `IDisposable`. Use `using` statements or call `Dispose()` explicitly:

```csharp
model.Dispose();
```

## Generation Options

| Field | Default | Description |
|-------|---------|-------------|
| `MaxTokens` | 256 | Maximum tokens to generate |
| `Temperature` | 1.0 | Sampling temperature |
| `TopK` | 50 | Top-k sampling |
| `RepeatPenalty` | 1.1 | Repetition penalty |
| `RepeatLastN` | 64 | Window for repetition penalty |

## Unity

OxBitNet targets `netstandard2.1` for Unity 2021.2+ compatibility. Place the native library in your Unity project's `Plugins` folder — Unity's plugin system handles native loading automatically.

## Running the Example

```bash
cd packages/rust
cargo build -p oxbitnet-ffi --release
cd crates/oxbitnet-csharp
dotnet run --project examples/ChatConsole -- /path/to/model.gguf
```

## License

MIT
