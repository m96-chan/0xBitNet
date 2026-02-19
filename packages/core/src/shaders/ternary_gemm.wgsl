// Ternary GEMM: batched matrix multiply for prompt processing
// C[M,N] = TernaryWeights[M,K] × Input[K,N]
//
// Weight packing (I2_S / llama.cpp BitNet fork):
//   4 ternary values per byte, MSB-first: bits[7:6]=elem0, [5:4]=elem1, [3:2]=elem2, [1:0]=elem3
//   code mapping: {0=-1, 1=0, 2=+1, 3=0(unused)}
//   16 values per u32 (little-endian)
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
@group(0) @binding(4) var<uniform> input_scale: f32;
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
        let packed_col = global_k / 16u;
        let elem_idx = global_k % 16u;
        let packed = weights[global_row * params.K_packed + packed_col];
        // I2_S packing: within each byte, MSB pair is first element
        let byte_idx = elem_idx >> 2u;
        let pair_idx = 3u - (elem_idx & 3u);
        let code = (packed >> (byte_idx * 8u + pair_idx * 2u)) & 3u;
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
        x_val = input[global_k * params.N + global_col];
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
    let scale = scales[global_row] * input_scale;
    for (var tn = 0u; tn < THREAD_TILE_N; tn++) {
      let global_col = wg_col + tid_n * THREAD_TILE_N + tn;
      if (global_col >= params.N) { continue; }
      output[global_row * params.N + global_col] = f32(acc[tm * THREAD_TILE_N + tn]) * scale;
    }
  }
}
