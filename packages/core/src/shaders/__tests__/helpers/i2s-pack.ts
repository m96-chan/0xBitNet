/**
 * I2_S packing/unpacking for ternary weights.
 *
 * Format (Eddie-Wang1120 llama.cpp fork):
 *   128-element block interleaving. Each 32-byte block stores 128 elements
 *   in 4 groups of 32. Byte[gp] within a block stores:
 *     bits[7:6] = group0 (offset 0*32 + gp)
 *     bits[5:4] = group1 (offset 1*32 + gp)
 *     bits[3:2] = group2 (offset 2*32 + gp)
 *     bits[1:0] = group3 (offset 3*32 + gp)
 *   Code mapping: {0=-1, 1=0, 2=+1}
 */

/** Map ternary value {-1, 0, +1} to 2-bit code {0, 1, 2} */
function encode(val: number): number {
  return val + 1; // -1→0, 0→1, +1→2
}

/** Map 2-bit code {0, 1, 2} to ternary value {-1, 0, +1} */
function decode(code: number): number {
  return code - 1; // 0→-1, 1→0, 2→+1
}

/**
 * Pack a flat array of ternary values into I2_S block-interleaved u32 format.
 * @param values  Flat array of {-1, 0, +1} values, length = K (padded to 128)
 * @param K       Logical element count (before padding)
 * @returns Uint32Array of packed u32s (K_packed = ceil(K/16) u32s)
 */
export function packI2S(values: number[], K: number): Uint32Array {
  const K128 = Math.ceil(K / 128) * 128;
  const padded = new Array(K128).fill(0);
  for (let i = 0; i < Math.min(values.length, K); i++) {
    padded[i] = values[i];
  }

  const numBlocks = K128 / 128;
  const bytes = new Uint8Array(numBlocks * 32);

  for (let block = 0; block < numBlocks; block++) {
    for (let gp = 0; gp < 32; gp++) {
      const g0 = encode(padded[block * 128 + 0 * 32 + gp]);
      const g1 = encode(padded[block * 128 + 1 * 32 + gp]);
      const g2 = encode(padded[block * 128 + 2 * 32 + gp]);
      const g3 = encode(padded[block * 128 + 3 * 32 + gp]);
      // bits[7:6]=g0, bits[5:4]=g1, bits[3:2]=g2, bits[1:0]=g3
      bytes[block * 32 + gp] = (g0 << 6) | (g1 << 4) | (g2 << 2) | g3;
    }
  }

  // Convert bytes to u32 (little-endian)
  const u32s = new Uint32Array(bytes.length / 4);
  const dv = new DataView(bytes.buffer);
  for (let i = 0; i < u32s.length; i++) {
    u32s[i] = dv.getUint32(i * 4, true); // little-endian
  }
  return u32s;
}

/**
 * Unpack I2_S block-interleaved u32 format back to ternary values.
 * Uses the SAME formula as the WGSL shader:
 *   block = k / 128, pos = k % 128, group = pos / 32, gp = pos % 32
 *   u32_idx = block * 8 + gp / 4
 *   shift = (gp % 4) * 8 + (6 - 2 * group)
 *   code = (packed >> shift) & 3
 *   w = code - 1
 */
export function unpackI2S(u32s: Uint32Array, K: number): number[] {
  const result: number[] = [];
  for (let k = 0; k < K; k++) {
    const block = Math.floor(k / 128);
    const pos = k % 128;
    const group = Math.floor(pos / 32);
    const gp = pos % 32;
    const u32Idx = block * 8 + Math.floor(gp / 4);
    const byteInU32 = gp % 4;
    const shift = byteInU32 * 8 + (6 - 2 * group);
    const packed = u32s[u32Idx];
    const code = (packed >>> shift) & 3;
    result.push(code - 1);
  }
  return result;
}

/**
 * Pack a weight matrix [M, K] of ternary values.
 * Each row is independently packed.
 * @returns Uint32Array of size M * K_packed
 */
export function packI2SMatrix(
  rows: number[][],
  M: number,
  K: number
): Uint32Array {
  const kPacked = Math.ceil(K / 16);
  const result = new Uint32Array(M * kPacked);
  for (let m = 0; m < M; m++) {
    const rowPacked = packI2S(rows[m], K);
    result.set(rowPacked, m * kPacked);
  }
  return result;
}

/**
 * Unpack an entire weight matrix.
 */
export function unpackI2SMatrix(
  u32s: Uint32Array,
  M: number,
  K: number
): number[][] {
  const kPacked = Math.ceil(K / 16);
  const result: number[][] = [];
  for (let m = 0; m < M; m++) {
    const row = u32s.slice(m * kPacked, (m + 1) * kPacked);
    result.push(unpackI2S(row, K));
  }
  return result;
}
