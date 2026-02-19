// Numerically stable softmax
//
// For each row:
//   1. Find max value (for numerical stability)
//   2. Compute sum of exp(x - max)
//   3. Normalize: out[i] = exp(x[i] - max) / sum
//
// Layout:
//   input:  [N, D] f32
//   output: [N, D] f32

struct Params {
  N: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_val: array<f32, 256>;

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

  // Pass 1: Find max
  var local_max: f32 = -3.402823e+38; // -FLT_MAX
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_max = max(local_max, input[row_offset + col]);
  }

  shared_val[tid] = local_max;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_val[tid] = max(shared_val[tid], shared_val[tid + stride]);
    }
    workgroupBarrier();
  }

  let row_max = shared_val[0];
  workgroupBarrier();

  // Pass 2: Sum of exp(x - max)
  var local_sum: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_sum += exp(input[row_offset + col] - row_max);
  }

  shared_val[tid] = local_sum;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_val[tid] += shared_val[tid + stride];
    }
    workgroupBarrier();
  }

  let inv_sum = 1.0 / shared_val[0];
  workgroupBarrier();

  // Pass 3: Normalize
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    output[row_offset + col] = exp(input[row_offset + col] - row_max) * inv_sum;
  }
}
