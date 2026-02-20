# oxbitnet

Run [BitNet b1.58](https://github.com/microsoft/BitNet) ternary LLMs with GPU acceleration (wgpu).

Python bindings for [0xBitNet](https://github.com/m96-chan/0xBitNet) — also available as [`0xbitnet`](https://www.npmjs.com/package/0xbitnet) (npm) and [`oxbitnet`](https://crates.io/crates/oxbitnet) (Rust).

## Install

```bash
pip install oxbitnet
```

## Quick Start

```python
from oxbitnet import BitNet

model = BitNet.load_sync("model.gguf")

# Chat with streaming output
model.chat(
    [("system", "You are a helpful assistant."), ("user", "Hello!")],
    on_token=lambda t: print(t, end="", flush=True),
    temperature=0.7,
    top_k=40,
)

model.dispose()
```

## API

| Method | Description |
|--------|-------------|
| `BitNet.load_sync(source)` | Load a GGUF model from URL or path |
| `model.chat(messages, on_token, ...)` | Chat with template + streaming callback |
| `model.generate(prompt, on_token, ...)` | Generate with streaming callback |
| `model.generate_sync(prompt, ...)` | Generate, return full string |
| `model.generate_tokens_sync(prompt, ...)` | Generate, return list of token strings |
| `model.dispose()` | Release GPU resources |

### Parameters

All generate methods accept:
- `max_tokens` (default: 256)
- `temperature` (default: 1.0)
- `top_k` (default: 50)
- `repeat_penalty` (default: 1.1)

## License

MIT
