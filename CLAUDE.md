# CLAUDE.md

## Project

0xBitNet — Run BitNet b1.58 ternary LLMs in the browser with WebGPU.
Monorepo: npm workspaces + Turborepo, tsup (core), Vite (examples).

## Past Mistakes

### GitHub Actions: turbo is a local binary
`npm ci` installs turbo locally. Use `npx turbo` or `npm run build/lint` in CI — bare `turbo` is not on PATH.

### Vite: set base for subdirectory deploys
When deploying to a subdirectory (e.g. `/chat/`, `/tldr/`), set `base: "./"` in vite.config.ts. Otherwise assets resolve from root and 404.

### Binary constants: double-check endianness
GGUF magic "GGUF" as little-endian uint32 is `0x46554747`, NOT `0x46475547`. When writing magic/constant bytes, always verify the byte order manually:
- Write out each ASCII char as hex: G=0x47, G=0x47, U=0x55, F=0x46
- Little-endian uint32 = bytes[0] | bytes[1]<<8 | bytes[2]<<16 | bytes[3]<<24

### Cache API: not suitable for large files
Cache API fails on large blobs (1GB+). Use IndexedDB for model caching instead.

### GGML types: BitNet fork uses non-standard type numbers
Microsoft's BitNet uses a fork of llama.cpp (Eddie-Wang1120/llama.cpp) with custom types.
I2_S is type **36** (not 27). Standard ggml type 27 is I64. Always verify type numbers
against the actual fork's ggml.h, not upstream.

### GGUF: always inspect actual files before assuming structure
Download header bytes and parse. Don't assume tensor names or metadata keys.
- Metadata prefix = `general.architecture` value (e.g. "bitnet-b1.58"), NOT "llama"/"bitnet"
- BitNet 2B-4T has `attn_sub_norm`/`ffn_sub_norm` (shared per-block norms), not per-projection norms
- I2_S weights are ALREADY packed: 4 ternary values per byte (2 bits each), 16 per u32.
  Upload raw bytes directly — do NOT repack. The data is NOT unpacked int8 {-1,0,1}.
- `token_embd.weight` is F16 (type=1), not F32
- No `output.weight` tensor → tied embeddings (tieWordEmbeddings=true)
