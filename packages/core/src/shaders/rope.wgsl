// Rotary Position Embeddings (RoPE)
//
// For each pair (x[2i], x[2i+1]) at position `pos`:
//   theta = pos * base^(-2i/D)
//   out[2i]   = x[2i]   * cos(theta) - x[2i+1] * sin(theta)
//   out[2i+1] = x[2i]   * sin(theta) + x[2i+1] * cos(theta)
//
// Layout:
//   input:  [N, num_heads, head_dim] f32
//   output: [N, num_heads, head_dim] f32
//   Dispatched per (token, head, pair)

struct Params {
  N: u32,          // sequence length
  num_heads: u32,
  head_dim: u32,
  pos_offset: u32, // starting position (for KV-cache continuation)
  theta_base: f32, // default 10000.0 or 500000.0
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let half_dim = params.head_dim / 2u;
  let total_pairs = params.N * params.num_heads * half_dim;

  let pair_idx = gid.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  // Decompose linear index into (token, head, dim_pair)
  let dim_pair = pair_idx % half_dim;
  let remainder = pair_idx / half_dim;
  let head = remainder % params.num_heads;
  let token = remainder / params.num_heads;

  let pos = f32(token + params.pos_offset);
  let freq_exp = -2.0 * f32(dim_pair) / f32(params.head_dim);
  let theta = pos * pow(params.theta_base, freq_exp);

  let cos_theta = cos(theta);
  let sin_theta = sin(theta);

  let base_idx = (token * params.num_heads + head) * params.head_dim + dim_pair * 2u;
  let x0 = input[base_idx];
  let x1 = input[base_idx + 1u];

  output[base_idx]      = x0 * cos_theta - x1 * sin_theta;
  output[base_idx + 1u] = x0 * sin_theta + x1 * cos_theta;
}
