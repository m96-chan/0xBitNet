// Ternary GEMM: batched matrix multiply for prompt processing
// Output[N,M] = Input[N,K] × TernaryWeights[M,K]^T
//
// Weight packing (I2_S / Eddie-Wang1120 llama.cpp fork):
//   128-element block interleaving. Each 32-byte block stores 128 elements
//   in 4 groups of 32. Byte[gp] within a block stores:
//     bits[7:6] = group0 (offset 0*32+gp), bits[5:4] = group1 (offset 1*32+gp)
//     bits[3:2] = group2 (offset 2*32+gp), bits[1:0] = group3 (offset 3*32+gp)
//   code mapping: {0=-1, 1=0, 2=+1}
// Input: int8 activations stored as i32
// Output: f32 (dequantized)
//
// 2D tiling: 16×16 workgroup, 4×4 per-thread output tile

struct Params {
  M: u32,        // output rows (weight rows)
  N: u32,        // output cols (batch / seq_len)
  K: u32,        // inner dimension (unpacked)
  K_packed: u32,  // K / 16
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<i32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> input_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const TILE_M: u32 = 64u;  // rows per workgroup
const TILE_N: u32 = 64u;  // cols per workgroup
const TILE_K: u32 = 32u;  // K-tile for shared memory (unpacked units)
const THREADS_M: u32 = 16u;
const THREADS_N: u32 = 16u;
const THREAD_TILE_M: u32 = 4u; // TILE_M / THREADS_M
const THREAD_TILE_N: u32 = 4u; // TILE_N / THREADS_N

var<workgroup> shared_w: array<i32, 2048>; // TILE_M × TILE_K
var<workgroup> shared_x: array<i32, 2048>; // TILE_K × TILE_N

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let wg_row = wg_id.x * TILE_M;
  let wg_col = wg_id.y * TILE_N;
  let tid_m = local_id.x;
  let tid_n = local_id.y;

  // Per-thread accumulators (4×4 tile)
  var acc: array<i32, 16>; // THREAD_TILE_M × THREAD_TILE_N
  for (var i = 0u; i < 16u; i++) {
    acc[i] = 0;
  }

  // Loop over K in tiles
  let k_tiles = (params.K + TILE_K - 1u) / TILE_K;
  for (var kt = 0u; kt < k_tiles; kt++) {
    let k_base = kt * TILE_K;

    // Cooperatively load weight tile into shared memory
    let linear_id = tid_m * THREADS_N + tid_n;
    let load_count = (TILE_M * TILE_K) / (THREADS_M * THREADS_N);
    for (var ld = 0u; ld < load_count; ld++) {
      let idx = linear_id + ld * (THREADS_M * THREADS_N);
      let local_row = idx / TILE_K;
      let local_col = idx % TILE_K;
      let global_row = wg_row + local_row;
      let global_k = k_base + local_col;

      var w_val: i32 = 0;
      if (global_row < params.M && global_k < params.K) {
        // I2_S 128-element block interleaving
        let block = global_k / 128u;
        let pos = global_k % 128u;
        let group = pos / 32u;
        let gp = pos % 32u;
        let u32_idx = block * 8u + gp / 4u;
        let byte_in_u32 = gp % 4u;
        let shift = byte_in_u32 * 8u + (6u - 2u * group);
        let packed = weights[global_row * params.K_packed + u32_idx];
        let code = (packed >> shift) & 3u;
        w_val = i32(code) - 1;
      }
      shared_w[local_row * TILE_K + local_col] = w_val;
    }

    // Cooperatively load input tile into shared memory
    let load_count_x = (TILE_K * TILE_N) / (THREADS_M * THREADS_N);
    for (var ld = 0u; ld < load_count_x; ld++) {
      let idx = linear_id + ld * (THREADS_M * THREADS_N);
      let local_k = idx / TILE_N;
      let local_col = idx % TILE_N;
      let global_k = k_base + local_k;
      let global_col = wg_col + local_col;

      var x_val: i32 = 0;
      if (global_k < params.K && global_col < params.N) {
        x_val = input[global_col * params.K + global_k];
      }
      shared_x[local_k * TILE_N + local_col] = x_val;
    }

    workgroupBarrier();

    // Compute per-thread 4×4 accumulation
    for (var k = 0u; k < TILE_K; k++) {
      for (var tm = 0u; tm < THREAD_TILE_M; tm++) {
        let w = shared_w[(tid_m * THREAD_TILE_M + tm) * TILE_K + k];
        for (var tn = 0u; tn < THREAD_TILE_N; tn++) {
          let x = shared_x[k * TILE_N + tid_n * THREAD_TILE_N + tn];
          acc[tm * THREAD_TILE_N + tn] += w * x;
        }
      }
    }

    workgroupBarrier();
  }

  // Write results with dequantization
  for (var tm = 0u; tm < THREAD_TILE_M; tm++) {
    let global_row = wg_row + tid_m * THREAD_TILE_M + tm;
    if (global_row >= params.M) { continue; }
    let w_scale = scales[global_row];
    for (var tn = 0u; tn < THREAD_TILE_N; tn++) {
      let global_col = wg_col + tid_n * THREAD_TILE_N + tn;
      if (global_col >= params.N) { continue; }
      let scale = w_scale * input_scales[global_col];
      output[global_col * params.M + global_row] = f32(acc[tm * THREAD_TILE_N + tn]) * scale;
    }
  }
}
