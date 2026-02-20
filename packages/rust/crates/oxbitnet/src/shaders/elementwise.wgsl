// Elementwise operations: add, multiply
//
// Used for residual connections and gating.
//
// Layout:
//   a:      [N] f32
//   b:      [N] f32
//   output: [N] f32

struct Params {
  N: u32,
  op: u32,  // 0 = add, 1 = multiply
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N) {
    return;
  }

  if (params.op == 0u) {
    output[idx] = a[idx] + b[idx];
  } else {
    output[idx] = a[idx] * b[idx];
  }
}
