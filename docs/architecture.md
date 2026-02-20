# Architecture

## Data Flow

```
                    ┌─────────────┐
                    │  GGUF File  │
                    │  (URL/IDB)  │
                    └──────┬──────┘
                           │ fetch / IndexedDB cache
                           ▼
                    ┌─────────────┐
                    │ GGUF Parser │  Parse header, metadata, tensor offsets
                    └──────┬──────┘
                           │ config + raw weight bytes
                           ▼
                    ┌─────────────┐
                    │Weight Loader│  Upload to GPU buffers, shard if needed
                    └──────┬──────┘
                           │ WeightStore (GPU buffers)
                           ▼
         ┌─────────────────────────────────┐
         │          BitNetModel            │
         │  ┌───────┐  ┌──────┐  ┌──────┐ │
         │  │Embed   │→│Layers│→│LM Head││
         │  │(F16)   │  │(×N)  │  │      ││
         │  └───────┘  └──┬───┘  └──────┘ │
         └────────────────┼────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         ┌────────┐ ┌──────────┐ ┌────────┐
         │  Attn  │ │BitLinear │ │  FFN   │
         │ (RoPE) │ │(Ternary) │ │(ReLU²) │
         └────────┘ └──────────┘ └────────┘
                          │
                          ▼
                    ┌───────────┐
                    │  Logits   │→ CPU readback → sample → token string
                    └───────────┘
```

## Project Layout

```
0xbitnet/
├── packages/core/                  # NPM package: 0xbitnet
│   └── src/
│       ├── index.ts                # Public API (BitNet class, re-exports)
│       ├── types.ts                # All TypeScript interfaces
│       │
│       ├── gpu/
│       │   ├── device.ts           # WebGPU init (initGPU, GPUDeviceError)
│       │   └── buffer-pool.ts      # GPU buffer recycling pool
│       │
│       ├── model/
│       │   ├── config.ts           # Predefined model configs
│       │   ├── loader.ts           # Fetch, cache (IndexedDB), parse, upload
│       │   ├── gguf.ts             # GGUF binary format parser
│       │   ├── safetensors.ts      # Safetensors format parser
│       │   └── weights.ts          # WeightStore (GPU buffer management)
│       │
│       ├── nn/
│       │   ├── model.ts            # BitNetModel (transformer forward pass)
│       │   ├── attention.ts        # Multi-head attention with RoPE
│       │   ├── bitlinear.ts        # BitLinear layer (ternary matmul)
│       │   ├── embedding.ts        # Token embedding (F16)
│       │   ├── norm.ts             # RMSNorm
│       │   └── ffn.ts              # Feed-forward network
│       │
│       ├── shaders/                # WGSL compute shaders
│       │   ├── matmul_ternary.wgsl # I2_S ternary matrix multiply
│       │   ├── rmsnorm.wgsl        # RMS normalization
│       │   ├── rope.wgsl           # Rotary position embeddings
│       │   ├── activation.wgsl     # ReLU², SiLU activations
│       │   └── ...                 # Other compute kernels
│       │
│       ├── tokenizer/
│       │   └── tokenizer.ts        # BPE tokenizer + chat template
│       │
│       └── worker/                 # Worker thread support
│           └── worker.ts           # Worker message handlers
│
├── examples/
│   ├── web-chat/                   # Chat app demo (Vite)
│   └── tl-dr-widget/              # Offline TL;DR widget demo (Vite)
│
└── docs/                           # Documentation
```

## BitLinear Pipeline

Each linear layer in BitNet uses ternary {-1, 0, +1} weights instead of floating-point. The BitLinear pipeline per layer:

```
Input (F32)
    │
    ▼
┌──────────┐
│ RMSNorm  │  Sub-norm: normalize before quantization
└────┬─────┘
     │
     ▼
┌──────────────────┐
│ Activation Quant │  Quantize input to INT8: round(clip(x * 127/absmax))
└────┬─────────────┘
     │  INT8 activations + scale factor (absmax / 127)
     ▼
┌──────────────────┐
│ Ternary MatMul   │  INT8 × {-1,0,+1} → INT32 accumulator
│ (WGSL kernel)    │  Each workgroup processes a tile of the output
└────┬─────────────┘
     │  INT32 accumulator
     ▼
┌──────────────────┐
│ Dequantize       │  output = acc × weight_scale × (absmax / 127)
└────┬─────────────┘
     │
     ▼
Output (F32)
```

The dequantization combines:
- **weight_scale**: per-tensor scale stored in the I2_S trailing 32 bytes
- **act_scale**: `absmax / 127`, the inverse of the input quantization factor

## I2_S Ternary Format

I2_S (2-bit integer, signed) is the ternary weight format used by Microsoft's BitNet GGUF files. Each weight value is one of {-1, 0, +1}, encoded as 2 bits.

### Block-Interleaved Packing

I2_S uses **128-element block interleaving**, not sequential packing. Each 32-byte block stores 128 ternary values organized into 4 groups of 32:

```
Block (32 bytes = 128 elements):
  Byte[i] stores 4 elements from different groups:
    bits[7:6] → group 0: element at offset i
    bits[5:4] → group 1: element at offset 32 + i
    bits[3:2] → group 2: element at offset 64 + i
    bits[1:0] → group 3: element at offset 96 + i
```

2-bit encoding: `0b00 = 0`, `0b01 = +1`, `0b10 = -1`, `0b11 = unused`

### Per-Tensor Scale

The last 32 bytes of each I2_S tensor contain the per-tensor `weight_scale` as a float32 value replicated 8 times. Total byte size: `ceil(numElements / 4) + 32`.

### Indexing

To extract element at logical index `k`:

```
block = k / 128
pos   = k % 128
group = pos / 32
gp    = pos % 32
u32_idx = block * 8 + gp / 4
shift   = (gp % 4) * 8 + (6 - 2 * group)
value   = (packed[u32_idx] >> shift) & 0x3
```

## Weight Loading Pipeline

```
1. Fetch       URL → ArrayBuffer (with streaming progress)
               ↕ IndexedDB cache (auto read/write)

2. Parse       ArrayBuffer → GGUFFile
               Extract: header, metadata, tensor info, data offset

3. Config      GGUF metadata → ModelConfig
               Auto-detect: architecture, head counts, tied embeddings

4. Upload      For each tensor:
               ├─ I2_S  → upload raw packed bytes + extract scale
               ├─ F16   → upload as-is (embedding) or convert to F32
               └─ Other → upload directly
               Shard large tensors that exceed maxStorageBufferBindingSize

5. Build       ModelConfig + WeightStore → BitNetModel
               Create compute pipelines, bind groups, KV cache
```

## Performance Optimizations

### BufferPool

GPU buffers are recycled through a `BufferPool` instead of creating and destroying buffers each forward pass. Buffers are matched by size and returned to the pool after use.

### Background Caching

In browser environments, models are cached in IndexedDB after the first download. Subsequent loads skip the network entirely and read directly from the local store.

### Pre-allocated Uniforms

Uniform buffers for shader parameters (dimensions, head counts, sequence positions) are created once and reused, avoiding per-dispatch allocation.

### Min-Heap Top-K

Token sampling uses an O(V) min-heap for top-K selection instead of O(V log V) sorting the entire vocabulary.

### F16 Embeddings

The embedding table is kept as F16 on the GPU (decoded via `unpack2x16float` in shaders). This halves the buffer size compared to F32, which can avoid hitting `maxStorageBufferBindingSize` limits on some GPUs.

### Weight Sharding

Tensors that exceed the GPU's `maxStorageBufferBindingSize` are automatically split into shards. The matmul kernel processes each shard and accumulates results.
