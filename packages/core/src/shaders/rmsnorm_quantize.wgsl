// Fused RMSNorm + Quantize: avoids intermediate normed buffer
//
// Three-pass within one dispatch:
//   Pass 1: sum of squares → rms (workgroup reduction)
//   Pass 2: normed values → absmax (workgroup reduction)
//   Pass 3: normalize + quantize → output
//
// Layout:
//   input:  [N, D] f32          (raw activations)
//   weight: [D]    f32          (RMSNorm scale)
//   output: [N, D] i32          (quantized int8 as i32)
//   scales: [N]    f32          (per-token absmax / 127)

struct Params {
  N: u32,
  D: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared: array<f32, 256>;

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
  let off = row * params.D;

  // Pass 1: sum of squares → rms
  var ss: f32 = 0.0;
  for (var c = tid; c < params.D; c += WORKGROUP_SIZE) {
    let v = input[off + c];
    ss += v * v;
  }
  shared[tid] = ss;
  workgroupBarrier();

  for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    workgroupBarrier();
  }

  let rms = inverseSqrt(shared[0] / f32(params.D) + params.eps);

  // Barrier before reusing shared memory for pass 2
  workgroupBarrier();

  // Pass 2: normed values → absmax
  var mx: f32 = 0.0;
  for (var c = tid; c < params.D; c += WORKGROUP_SIZE) {
    mx = max(mx, abs(input[off + c] * rms * weight[c]));
  }
  shared[tid] = mx;
  workgroupBarrier();

  for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (tid < s) {
      shared[tid] = max(shared[tid], shared[tid + s]);
    }
    workgroupBarrier();
  }

  let absmax = shared[0];
  let inv_s = select(127.0 / absmax, 0.0, absmax == 0.0);

  if (tid == 0u) {
    scales[row] = select(absmax / 127.0, 1.0, absmax == 0.0);
  }

  // Pass 3: normalize + quantize
  for (var c = tid; c < params.D; c += WORKGROUP_SIZE) {
    let n = input[off + c] * rms * weight[c];
    output[off + c] = clamp(i32(round(n * inv_s)), -127, 127);
  }
}
