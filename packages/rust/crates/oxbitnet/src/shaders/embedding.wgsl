// Token embedding lookup (F16 on GPU)
//
// For each token ID, copy the corresponding row from the embedding table.
// Embedding table is stored as packed F16 pairs (two f16 values per u32)
// to avoid exceeding maxStorageBufferBindingSize on most GPUs.
//
// Layout:
//   token_ids:  [N]          u32
//   embed_table: [V * D / 2] u32  (packed f16 pairs)
//   output:     [N, D]       f32

struct Params {
  N: u32,  // number of tokens
  D: u32,  // embedding dimension
  V: u32,  // vocab size
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> embed_table: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  let total = params.N * params.D;
  if (idx >= total) {
    return;
  }

  let token = idx / params.D;
  let dim = idx % params.D;
  let token_id = token_ids[token];

  // Bounds check: treat out-of-vocab as zero
  if (token_id < params.V) {
    let flat = token_id * params.D + dim;
    let packed = embed_table[flat / 2u];
    let pair = unpack2x16float(packed);
    output[idx] = select(pair.x, pair.y, (flat & 1u) == 1u);
  } else {
    output[idx] = 0.0;
  }
}
