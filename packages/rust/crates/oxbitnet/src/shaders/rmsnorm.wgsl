// RMSNorm: x_i * w_i / sqrt(mean(x²) + eps)
//
// Two-pass within one dispatch:
//   1. Compute sum of squares (workgroup reduction)
//   2. Normalize: x_i * w_i * rsqrt(mean_sq + eps)
//
// Layout:
//   input:  [N, D] f32
//   weight: [D]    f32 (learnable scale)
//   output: [N, D] f32

struct Params {
  N: u32,
  D: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.N) {
    return;
  }

  let tid = local_id.x;
  let row_offset = row * params.D;

  // Pass 1: Sum of squares
  var local_sum: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    let val = input[row_offset + col];
    local_sum += val * val;
  }

  shared_sum[tid] = local_sum;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    workgroupBarrier();
  }

  let rms = inverseSqrt(shared_sum[0] / f32(params.D) + params.eps);

  // Pass 2: Normalize
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    output[row_offset + col] = input[row_offset + col] * rms * weight[col];
  }
}
