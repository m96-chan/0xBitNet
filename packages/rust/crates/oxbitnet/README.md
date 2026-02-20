# oxbitnet

Run [BitNet b1.58](https://github.com/microsoft/BitNet) ternary LLMs with [wgpu](https://wgpu.rs/).

Part of [0xBitNet](https://github.com/m96-chan/0xBitNet) — also available as [`0xbitnet`](https://www.npmjs.com/package/0xbitnet) (npm) and [`oxbitnet`](https://pypi.org/project/oxbitnet/) (Python).

## Quick Start

```rust
use oxbitnet::BitNet;
use futures::StreamExt;

#[tokio::main]
async fn main() -> oxbitnet::Result<()> {
    let mut model = BitNet::load("model.gguf", Default::default()).await?;

    let options = oxbitnet::GenerateOptions {
        max_tokens: 256,
        temperature: 0.7,
        top_k: 40,
        repeat_penalty: 1.1,
        ..Default::default()
    };

    let mut stream = model.generate_chat(
        &[
            oxbitnet::ChatMessage { role: "system".into(), content: "You are a helpful assistant.".into() },
            oxbitnet::ChatMessage { role: "user".into(), content: "Hello!".into() },
        ],
        options,
    );

    while let Some(token) = stream.next().await {
        print!("{token}");
    }

    model.dispose();
    Ok(())
}
```

## Features

- **Native wgpu** — Vulkan, Metal, DX12 backends automatically
- **GGUF loading** — Handles I2_S ternary packing (Microsoft BitNet fork)
- **Streaming** — Token-by-token via `impl Stream<Item = String>`
- **Chat templates** — Built-in LLaMA 3 chat formatting
- **Disk caching** — Models cached at `~/.cache/.0xbitnet/`

## Example

Interactive chat CLI:

```sh
cargo run --example chat --release
cargo run --example chat --release -- --url /path/to/model.gguf
```

## License

MIT
