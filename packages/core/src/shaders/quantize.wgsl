// Per-token absmax activation quantization: f32 → int8
//
// Two-pass approach:
//   Pass 1: Compute absmax per row (token)
//   Pass 2: Scale and round to [-127, 127]
//
// This shader combines both passes using workgroup reduction.
//
// Layout:
//   input:  [N, D] f32
//   output: [N, D] i32 (int8 stored as i32 for compute compatibility)
//   scales: [N]    f32 (per-token absmax / 127)

struct Params {
  N: u32,  // number of tokens
  D: u32,  // hidden dimension
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_max: array<f32, 256>;

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

  // Pass 1: Find absmax
  var local_max: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_max = max(local_max, abs(input[row_offset + col]));
  }

  shared_max[tid] = local_max;
  workgroupBarrier();

  // Reduction for max
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
    }
    workgroupBarrier();
  }

  let absmax = shared_max[0];
  let scale = select(absmax / 127.0, 1.0, absmax == 0.0);

  if (tid == 0u) {
    scales[row] = scale;
  }

  workgroupBarrier();

  // Pass 2: Quantize
  let inv_scale = select(127.0 / absmax, 0.0, absmax == 0.0);
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    let val = input[row_offset + col];
    let quantized = clamp(i32(round(val * inv_scale)), -127, 127);
    output[row_offset + col] = quantized;
  }
}
