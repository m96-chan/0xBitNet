// Ternary GEMV: packed ternary weights × int8 activations → i32 accumulator
//
// Weight packing: 16 ternary values per u32 (2 bits each)
//   code ∈ {0,1,2} → weight = code - 1 ∈ {-1,0,+1}
//   packed >> (2*i) & 3 gives the i-th code
//
// Layout:
//   weights: [M, K/16] u32  (packed ternary)
//   input:   [K]       i32  (int8 stored as i32)
//   scales:  [M]       f32  (per-row weight scale)
//   input_scale: f32         (activation absmax scale)
//   output:  [M]       f32
//
// Each workgroup processes one output row.
// Threads cooperatively reduce over the K dimension.

struct Params {
  M: u32,       // output rows
  K: u32,       // input dimension (unpacked)
  K_packed: u32, // K / 16
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<i32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> input_scale: f32;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sums: array<i32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.M) {
    return;
  }

  let tid = local_id.x;
  let row_offset = row * params.K_packed;

  var acc: i32 = 0;

  // Each thread processes a strided slice of packed columns
  for (var col = tid; col < params.K_packed; col += WORKGROUP_SIZE) {
    let packed = weights[row_offset + col];
    let base_k = col * 16u;

    // Unpack 16 ternary weights and dot with input
    for (var i = 0u; i < 16u; i++) {
      let k_idx = base_k + i;
      if (k_idx < params.K) {
        let code = (packed >> (2u * i)) & 3u;
        let w = i32(code) - 1;  // branchless: {0,1,2} → {-1,0,+1}
        acc += w * input[k_idx];
      }
    }
  }

  // Workgroup reduction
  shared_sums[tid] = acc;
  workgroupBarrier();

  // Tree reduction
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sums[tid] += shared_sums[tid + stride];
    }
    workgroupBarrier();
  }

  // Thread 0 writes the dequantized result
  if (tid == 0u) {
    let sum = f32(shared_sums[0]);
    output[row] = sum * scales[row] * input_scale;
  }
}
