// Standard f32 attention matmul kernels
//
// Two operations:
//   1. scores = Q @ K^T * scale    (score computation)
//   2. output = attn_weights @ V   (value aggregation)
//
// These use standard f32 matmul (not ternary) because Q,K,V are
// already projected through BitLinear and are f32 activations.

// ─── Kernel 1: Q @ K^T (score computation) ───
// Q:      [N, num_heads, head_dim]
// K:      [S, num_kv_heads, head_dim]  (S = total seq including cache)
// scores: [num_heads, N, S]

struct ScoreParams {
  N: u32,           // query seq length
  S: u32,           // key seq length (including cache)
  num_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
  scale: f32,       // 1/sqrt(head_dim)
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: ScoreParams;

@compute @workgroup_size(16, 16)
fn compute_scores(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  // gid.x = query position, gid.y = key position, gid.z = head
  let q_pos = gid.x;
  let k_pos = gid.y;
  let head = gid.z;

  if (q_pos >= params.N || k_pos >= params.S || head >= params.num_heads) {
    return;
  }

  // GQA: map attention head to KV head
  let kv_head = head / (params.num_heads / params.num_kv_heads);

  let q_offset = (q_pos * params.num_heads + head) * params.head_dim;
  let k_offset = (k_pos * params.num_kv_heads + kv_head) * params.head_dim;

  var dot: f32 = 0.0;
  for (var d = 0u; d < params.head_dim; d++) {
    dot += Q[q_offset + d] * K[k_offset + d];
  }

  // Causal mask: positions after query are -inf
  let is_causal = k_pos > q_pos + (params.S - params.N);
  let masked_score = select(dot * params.scale, -3.402823e+38, is_causal);

  let score_idx = (head * params.N + q_pos) * params.S + k_pos;
  scores[score_idx] = masked_score;
}

// ─── Kernel 2: Attention weights @ V ───
// attn:   [num_heads, N, S]
// V:      [S, num_kv_heads, head_dim]
// output: [N, num_heads, head_dim]

struct AttnVParams {
  N: u32,
  S: u32,
  num_heads: u32,
  num_kv_heads: u32,
  head_dim: u32,
}

@group(0) @binding(0) var<storage, read> attn: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> attn_output: array<f32>;
@group(0) @binding(3) var<uniform> attn_v_params: AttnVParams;

@compute @workgroup_size(256)
fn attn_v(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let total = attn_v_params.N * attn_v_params.num_heads * attn_v_params.head_dim;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }

  let d = idx % attn_v_params.head_dim;
  let remainder = idx / attn_v_params.head_dim;
  let head = remainder % attn_v_params.num_heads;
  let q_pos = remainder / attn_v_params.num_heads;

  let kv_head = head / (attn_v_params.num_heads / attn_v_params.num_kv_heads);

  var sum: f32 = 0.0;
  for (var s = 0u; s < attn_v_params.S; s++) {
    let a = attn[(head * attn_v_params.N + q_pos) * attn_v_params.S + s];
    let v = V[(s * attn_v_params.num_kv_heads + kv_head) * attn_v_params.head_dim + d];
    sum += a * v;
  }

  let out_idx = (q_pos * attn_v_params.num_heads + head) * attn_v_params.head_dim + d;
  attn_output[out_idx] = sum;
}
