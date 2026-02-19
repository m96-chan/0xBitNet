// Ternary GEMV: packed ternary weights × int8 activations → i32 accumulator
//
// Weight packing (I2_S / Eddie-Wang1120 llama.cpp fork):
//   128-element block interleaving for SIMD. Each 32-byte block stores 128 elements
//   in 4 groups of 32. Byte[gp] within a block stores:
//     bits[7:6] = element at group0 (offset 0*32 + gp)
//     bits[5:4] = element at group1 (offset 1*32 + gp)
//     bits[3:2] = element at group2 (offset 2*32 + gp)
//     bits[1:0] = element at group3 (offset 3*32 + gp)
//   code mapping: {0=-1, 1=0, 2=+1}
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

  // Each thread processes a strided slice of packed u32 columns
  for (var col = tid; col < params.K_packed; col += WORKGROUP_SIZE) {
    let packed = weights[row_offset + col];

    // I2_S block interleaving: 128 elements per 32-byte (8 u32) block
    let block = col / 8u;
    let base_gp = (col % 8u) * 4u;

    // Unpack 16 ternary weights from this u32 and dot with input
    for (var i = 0u; i < 16u; i++) {
      let byte_in_u32 = i / 4u;
      let group = i % 4u;
      let gp = base_gp + byte_in_u32;
      let k_idx = block * 128u + group * 32u + gp;
      if (k_idx < params.K) {
        let shift = byte_in_u32 * 8u + (6u - 2u * group);
        let code = (packed >> shift) & 3u;
        let w = i32(code) - 1;
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
