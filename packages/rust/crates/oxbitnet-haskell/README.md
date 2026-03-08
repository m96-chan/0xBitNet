# oxbitnet-haskell

Haskell bindings for [oxbitnet](https://crates.io/crates/oxbitnet) — run [BitNet b1.58](https://github.com/microsoft/BitNet) ternary LLMs with GPU acceleration (wgpu).

Part of [0xBitNet](https://github.com/m96-chan/0xBitNet).

## Build

First, build the native library:

```bash
cd packages/rust
cargo build -p oxbitnet-ffi --release
```

Produces `target/release/liboxbitnet_ffi.so` (Linux) / `.dylib` (macOS) / `oxbitnet_ffi.dll` (Windows).

Then build the Haskell package:

```bash
cd packages/rust/crates/oxbitnet-haskell
cabal build all \
  --extra-lib-dirs=../../target/release \
  --extra-include-dirs=../oxbitnet-ffi
```

## Quick Start

```haskell
import OxBitNet

main :: IO ()
main = withBitNet "model.gguf" defaultLoadOptions $ \model -> do
    -- Raw prompt
    generate model "Hello!" defaultGenerateOptions $ \token -> do
        putStr token
        return False  -- False = continue, True = stop

    -- Chat messages
    chat model [userMessage "Hello!"] defaultGenerateOptions $ \token -> do
        putStr token
        return False
```

## API

### Loading

```haskell
-- Bracket-based (recommended)
withBitNet "model.gguf" defaultLoadOptions $ \model -> do
    ...

-- Manual load/free
model <- loadBitNet "model.gguf" defaultLoadOptions
-- ...use model...
freeBitNet model

-- With progress callback
let opts = defaultLoadOptions
      { onProgress = Just $ \p ->
          putStrLn $ show (lpPhase p) ++ " " ++ show (lpFraction p * 100) ++ "%"
      }
withBitNet "model.gguf" opts $ \model -> ...
```

### Generation

```haskell
-- Raw prompt — tokens delivered via callback
generate model "Once upon a time" defaultGenerateOptions $ \token -> do
    putStr token
    return False  -- continue

-- With custom options
let opts = defaultGenerateOptions
      { maxTokens = 512, temperature = 0.7, topK = 40 }
generate model "Hello!" opts $ \token -> do
    putStr token
    return False

-- Stop early
generate model "Hello!" defaultGenerateOptions $ \token -> do
    putStr token
    return (token == "\n")  -- stop on newline
```

### Chat

```haskell
let messages =
      [ systemMessage "You are a helpful assistant."
      , userMessage "What is 2+2?"
      ]

chat model messages defaultGenerateOptions $ \token -> do
    putStr token
    return False
```

### Logger

```haskell
-- Install before loading any model (can only be called once)
setLogger Info $ \level msg ->
    putStrLn $ "[" ++ show level ++ "] " ++ msg
```

### Cleanup

`withBitNet` handles cleanup automatically via `bracket`. For manual management, use `loadBitNet` / `freeBitNet`. Calling `freeBitNet` multiple times is safe.

## Generation Options

| Field | Default | Description |
|-------|---------|-------------|
| `maxTokens` | 256 | Maximum tokens to generate |
| `temperature` | 1.0 | Sampling temperature |
| `topK` | 50 | Top-k sampling |
| `repeatPenalty` | 1.1 | Repetition penalty |
| `repeatLastN` | 64 | Window for repetition penalty |

## Exceptions

All errors are thrown as `OxBitNetException`:

- `LoadError String` — model failed to load
- `GenerateError String` — generation failed
- `Disposed` — attempted to use a freed model handle

## Running the Example

```bash
cd packages/rust
cargo build -p oxbitnet-ffi --release
cd crates/oxbitnet-haskell
cabal run oxbitnet-chat -- /path/to/model.gguf \
  --extra-lib-dirs=../../target/release
```

## License

MIT
