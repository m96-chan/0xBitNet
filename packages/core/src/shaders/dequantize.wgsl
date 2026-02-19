// Dequantize: i32 accumulator → f32
// output[i] = f32(input[i]) * weight_scale[row] * input_scale
//
// Used after ternary matmul when separate dequantization is needed.

struct Params {
  N: u32,  // number of elements
}

@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> weight_scale: f32;
@group(0) @binding(3) var<uniform> input_scale: f32;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N) {
    return;
  }

  output[idx] = f32(input[idx]) * weight_scale * input_scale;
}
