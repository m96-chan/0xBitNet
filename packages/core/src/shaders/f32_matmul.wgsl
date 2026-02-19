// F32 GEMV for tied-embedding LM head
// logits[n, v] = sum_d( hidden[n, d] * embed[v, d] )
//
// hidden: [N, D] f32 — final hidden states
// embed:  [V, D] f32 — embedding table (shared with LM head)
// output: [N, V] f32 — logits
//
// Each workgroup computes one (n, v) element.
// 256 threads cooperatively reduce over D.
// 2D dispatch: v = wg_id.x + wg_id.y * 65535  (V can exceed 65535)

struct Params {
  N: u32,
  V: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> embed: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  // Decode (n, v) from 2D dispatch
  let flat_id = wg_id.x + wg_id.y * 65535u;
  let n = flat_id / params.V;
  let v = flat_id % params.V;

  if (n >= params.N || v >= params.V) {
    return;
  }

  let tid = local_id.x;

  // Each thread accumulates a strided slice of D
  var acc: f32 = 0.0;
  let hidden_base = n * params.D;
  let embed_base = v * params.D;

  for (var d = tid; d < params.D; d += WG_SIZE) {
    acc += hidden[hidden_base + d] * embed[embed_base + d];
  }

  // Workgroup reduction
  shared_sums[tid] = acc;
  workgroupBarrier();

  for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
    }
    workgroupBarrier();
  }

  // Thread 0 writes the result
  if (tid == 0u) {
    output[n * params.V + v] = shared_sums[0];
  }
}
