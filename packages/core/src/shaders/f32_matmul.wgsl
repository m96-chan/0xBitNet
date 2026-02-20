// F32 GEMV for tied-embedding LM head (F16 embedding on GPU)
// logits[n, v] = sum_d( hidden[n, d] * embed[v, d] )
//
// hidden: [N, D] f32 — final hidden states
// embed:  [V * D / 2] u32 — embedding table stored as packed f16 pairs
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
@group(0) @binding(1) var<storage, read> embed: array<u32>;
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
  // Process pairs of dimensions for efficiency
  var acc: f32 = 0.0;
  let hidden_base = n * params.D;
  let embed_base = v * params.D;

  // Process two dimensions at a time using packed f16 pairs
  let D_half = params.D / 2u;
  for (var dh = tid; dh < D_half; dh += WG_SIZE) {
    let d = dh * 2u;
    let packed = embed[embed_base / 2u + dh];
    let pair = unpack2x16float(packed);
    acc += hidden[hidden_base + d] * pair.x;
    acc += hidden[hidden_base + d + 1u] * pair.y;
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
