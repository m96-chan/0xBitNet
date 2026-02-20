# oxbitnet-ffi

C FFI bindings for [oxbitnet](https://crates.io/crates/oxbitnet) — run [BitNet b1.58](https://github.com/microsoft/BitNet) ternary LLMs with GPU acceleration (wgpu).

Part of [0xBitNet](https://github.com/m96-chan/0xBitNet). Provides a stable C ABI for building language bindings (Java/Android, Swift/iOS, C#, Haskell, etc.).

## Build

```bash
cargo build -p oxbitnet-ffi --release
```

Produces:
- `target/release/liboxbitnet_ffi.so` (Linux) / `.dylib` (macOS) / `.dll` (Windows)
- `target/release/liboxbitnet_ffi.a` (static library)
- `crates/oxbitnet-ffi/oxbitnet.h` (auto-generated C header via cbindgen)

## Quick Start

```c
#include "oxbitnet.h"
#include <stdio.h>

static int32_t on_token(const char *token, uintptr_t len, void *userdata) {
    fwrite(token, 1, len, stdout);
    return 0; /* 0 = continue, non-zero = stop */
}

int main(void) {
    OxBitNet *model = oxbitnet_load("model.gguf", NULL);
    if (!model) {
        fprintf(stderr, "Error: %s\n", oxbitnet_error_message());
        return 1;
    }

    OxBitNetChatMessage messages[] = {
        { .role = "user", .content = "Hello!" },
    };
    OxBitNetGenerateOptions opts = oxbitnet_default_generate_options();

    oxbitnet_chat(model, messages, 1, &opts, on_token, NULL);
    printf("\n");

    oxbitnet_free(model);
    return 0;
}
```

Compile and run:

```bash
gcc chat.c -I. -L../../target/release -loxbitnet_ffi -o chat
LD_LIBRARY_PATH=../../target/release ./chat
```

## API

| Function | Description |
|----------|-------------|
| `oxbitnet_load(source, options)` | Load model from URL or path, returns opaque handle (NULL on error) |
| `oxbitnet_free(model)` | Free handle and release GPU resources |
| `oxbitnet_generate(model, prompt, opts, cb, ud)` | Generate from raw prompt with streaming callback |
| `oxbitnet_chat(model, msgs, n, opts, cb, ud)` | Generate from chat messages with streaming callback |
| `oxbitnet_error_message()` | Get last error message (thread-local, NULL if none) |
| `oxbitnet_set_logger(cb, ud, level)` | Install a logger callback (call before `oxbitnet_load`) |
| `oxbitnet_default_generate_options()` | Get default generation options |
| `oxbitnet_default_load_options()` | Get default load options |

### Token Callback

```c
typedef int32_t (*OxBitNetTokenFn)(const char *token, uintptr_t len, void *userdata);
```

Return 0 to continue generation, non-zero to stop early.

### Generation Options

| Field | Default | Description |
|-------|---------|-------------|
| `max_tokens` | 256 | Maximum tokens to generate |
| `temperature` | 1.0 | Sampling temperature |
| `top_k` | 50 | Top-k sampling |
| `repeat_penalty` | 1.1 | Repetition penalty |
| `repeat_last_n` | 64 | Window size for repetition penalty |

### Logger

```c
oxbitnet_set_logger(my_log_fn, NULL, 2 /* Info */);
```

Level values: 0=Trace, 1=Debug, 2=Info, 3=Warn, 4=Error. Must be called once before `oxbitnet_load`. Routes all internal `tracing` output through the callback.

### Thread Safety

Each `OxBitNet` handle owns its own tokio runtime. Handles must not be shared across threads without external synchronization. Error messages are thread-local.

## License

MIT
