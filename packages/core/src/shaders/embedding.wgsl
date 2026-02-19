// Token embedding lookup
//
// For each token ID, copy the corresponding row from the embedding table.
//
// Layout:
//   token_ids:  [N]      u32
//   embed_table: [V, D]  f32  (V = vocab_size, D = hidden_dim)
//   output:     [N, D]   f32

struct Params {
  N: u32,  // number of tokens
  D: u32,  // embedding dimension
  V: u32,  // vocab size
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> embed_table: array<f32>;
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
    output[idx] = embed_table[token_id * params.D + dim];
  } else {
    output[idx] = 0.0;
  }
}
