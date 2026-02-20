// Activation functions for BitNet FFN
//
// ReLU²: relu(x)² — used in official 2B-4T model
// SiLU:  x * sigmoid(x) — used in community models
//
// Layout:
//   input:  [N] f32
//   output: [N] f32

struct Params {
  N: u32,
  activation_type: u32,  // 0 = ReLU², 1 = SiLU
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N) {
    return;
  }

  let x = input[idx];

  if (params.activation_type == 0u) {
    // ReLU²: max(0, x)²
    let relu_x = max(0.0, x);
    output[idx] = relu_x * relu_x;
  } else {
    // SiLU: x * sigmoid(x)
    output[idx] = x / (1.0 + exp(-x));
  }
}
