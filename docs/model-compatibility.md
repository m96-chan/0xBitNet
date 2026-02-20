# Model Compatibility

## Tested Models

| Model | GGUF | Params | Activation | Tied Embeddings |
|-------|------|--------|------------|-----------------|
| [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | [ggml-model-i2_s.gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) | 2B | ReLU² | Yes |

Other BitNet-architecture models (e.g., bitnet_b1_58-large, Falcon3-Edge, Aramis-2B) may work via GGUF auto-detection but are not yet verified. See [#1](https://github.com/m96-chan/0xBitNet/issues/1) for progress on additional model support.

## GGUF Requirements

For a GGUF file to be loadable by 0xBitNet, it must meet these requirements:

### Required Metadata

The following GGUF metadata keys are used to derive the model config (prefixed by the `general.architecture` value):

| Key | Example | Description |
|-----|---------|-------------|
| `general.architecture` | `"bitnet-25"` | Architecture identifier (used as key prefix) |
| `{arch}.embedding_length` | `2560` | Hidden size |
| `{arch}.block_count` | `30` | Number of transformer layers |
| `{arch}.attention.head_count` | `20` | Number of attention heads |
| `{arch}.attention.head_count_kv` | `5` | Number of key-value heads (for GQA) |
| `{arch}.feed_forward_length` | `6912` | FFN intermediate size |

### Required Tokenizer Metadata

The tokenizer is extracted from GGUF metadata. These keys must be present:

| Key | Description |
|-----|-------------|
| `tokenizer.ggml.tokens` | Token vocabulary (string array) |
| `tokenizer.ggml.token_type` | Token type array |
| `tokenizer.ggml.merges` | BPE merge rules |

### Weight Types

| Tensor | Expected GGML Type | Notes |
|--------|-------------------|-------|
| `token_embd.weight` | F16 (type 1) | Embedding table |
| `blk.*.attn_q.weight` | I2_S (type 36) | Ternary weights |
| `blk.*.attn_k.weight` | I2_S (type 36) | Ternary weights |
| `blk.*.attn_v.weight` | I2_S (type 36) | Ternary weights |
| `blk.*.attn_output.weight` | I2_S (type 36) | Ternary weights |
| `blk.*.ffn_up.weight` | I2_S (type 36) | Ternary weights |
| `blk.*.ffn_down.weight` | I2_S (type 36) | Ternary weights |
| `blk.*.ffn_gate.weight` | I2_S (type 36) | Ternary weights (SiLU models only) |
| `blk.*.attn_norm.weight` | F16 (type 1) | RMSNorm |
| `blk.*.ffn_norm.weight` | F16 (type 1) | RMSNorm |
| `blk.*.attn_sub_norm.weight` | F16 (type 1) | BitNet sub-norm (2B-4T) |
| `blk.*.ffn_sub_norm.weight` | F16 (type 1) | BitNet sub-norm (2B-4T) |
| `output_norm.weight` | F16 (type 1) | Final RMSNorm |
| `output.weight` | I2_S (type 36) | LM head (absent = tied embeddings) |

**I2_S is type 36** from the Eddie-Wang1120/llama.cpp BitNet fork, not standard ggml type 27 (which is I64).

## Auto-Detection

0xBitNet automatically derives the model configuration from GGUF metadata:

1. **Architecture prefix** — Read from `general.architecture` (e.g., `"bitnet-25"`, `"llama"`, `"bitnet"`)
2. **Model dimensions** — Extracted from architecture-prefixed metadata keys
3. **Tied embeddings** — Detected by the absence of an `output.weight` tensor
4. **Activation function** — Official BitNet models (vocab > 100k or arch contains "bitnet") use ReLU²; others default to SiLU

The loader tries multiple key prefixes (`{arch}.`, `llama.`, `bitnet.`, `bitnet-25.`) to maximize compatibility.

## Known Limitations

### Single-File GGUF Only

0xBitNet currently supports only single-file GGUF models. Split GGUF files (e.g., `model-00001-of-00003.gguf`) are not supported.

### WebGPU Buffer Size Limits

WebGPU buffer sizes are constrained by `maxStorageBufferBindingSize` (typically 1–2 GB on most GPUs). Large tensors are automatically sharded, but total VRAM must fit the model plus KV cache.

### Storage Quota (Browser)

In browsers, IndexedDB caching is subject to storage quotas. A 2B model GGUF is typically ~700 MB. Browsers may prompt the user for persistent storage permission for large files. In non-browser environments, caching is skipped.

### No Safetensors Tokenizer

When loading Safetensors files (not GGUF), the tokenizer must be available as a separate `tokenizer.json` file in the same directory as the model file.

### Precision

All intermediate computations use F32. There is no F16 compute path (current WebGPU shader model limitations). This means compute is slower than native F16 would be, but accuracy is higher.
