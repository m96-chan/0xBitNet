# oxbitnet-java

Java/JNI bindings for [oxbitnet](https://crates.io/crates/oxbitnet) — run [BitNet b1.58](https://github.com/microsoft/BitNet) ternary LLMs with GPU acceleration (wgpu).

Part of [0xBitNet](https://github.com/m96-chan/0xBitNet).

## Build

```bash
cargo build -p oxbitnet-java --release
```

Produces `target/release/liboxbitnet_java.so` (Linux) / `.dylib` (macOS) / `oxbitnet_java.dll` (Windows).

## Quick Start

```java
import io.github.m96chan.oxbitnet.*;
import java.util.List;

try (BitNet model = BitNet.loadSync("model.gguf")) {
    // Raw prompt
    model.generate("Hello!", token -> {
        System.out.print(token);
        return true; // continue
    });

    // Chat messages
    model.chat(
        List.of(new ChatMessage("user", "Hello!")),
        token -> {
            System.out.print(token);
            return true;
        },
        new GenerateOptions().temperature(0.7f).maxTokens(256)
    );
}
```

## API

### Loading

```java
// Simple
BitNet model = BitNet.loadSync("model.gguf");

// With progress callback
BitNet model = BitNet.loadSync("model.gguf", new LoadOptions()
    .onProgress((phase, loaded, total, fraction) ->
        System.err.printf("[%s] %.1f%%\n", phase, fraction * 100)));
```

### Generation

```java
// Raw prompt
int tokens = model.generate("Once upon a time", token -> {
    System.out.print(token);
    return true; // return false to stop early
});

// With options
model.generate("Hello!", callback, new GenerateOptions()
    .temperature(0.7f)
    .maxTokens(512)
    .topK(40)
    .repeatPenalty(1.1f)
    .repeatLastN(64));
```

### Chat

```java
model.chat(
    List.of(
        new ChatMessage("system", "You are a helpful assistant."),
        new ChatMessage("user", "What is 2+2?")
    ),
    token -> {
        System.out.print(token);
        return true;
    }
);
```

### Cleanup

`BitNet` implements `AutoCloseable`:

```java
try (BitNet model = BitNet.loadSync("model.gguf")) {
    // use model...
} // automatically disposed
```

Or manually:

```java
model.dispose();
```

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
cargo build -p oxbitnet-java --release
cd crates/oxbitnet-java/examples
javac -cp ../java/src/main/java:. Chat.java
java -Djava.library.path=../../../../target/release -cp ../java/src/main/java:. Chat model.gguf "Hello!"
```

## License

MIT
