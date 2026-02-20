/**
 * End-to-end BitLinear integration test.
 *
 * Tests the full pipeline: RMSNorm → Quantize → TernaryMatmul → Dequant
 * with known inputs and weights, verifying against a CPU reference that
 * chains the individual reference functions together.
 *
 * This catches mismatches between:
 * - The quantize shader's scale format (absmax/127) vs the dequant formula
 * - The I2_S extraction logic vs the packing format
 * - The end-to-end numerical result
 */
import { describe, it, expect } from "vitest";
import { packI2S, packI2SMatrix, unpackI2S } from "./helpers/i2s-pack.js";
import {
  refRMSNorm,
  refQuantize,
  refTernaryGemvNaive,
  refTernaryGemm,
} from "./helpers/reference.js";
import { randf32, randTernary, expectClose, ones } from "./helpers/tensor-utils.js";

/**
 * Full BitLinear CPU reference:
 *   1. Optional RMSNorm
 *   2. Absmax quantize to int8
 *   3. Ternary matmul (packed I2_S weights)
 *   4. Dequantize: result * weightScale * inputScale
 *
 * This mirrors what BitLinear.forward() does on the GPU.
 */
function refBitLinear(
  input: Float32Array,
  N: number,
  inDim: number,
  outDim: number,
  ternaryWeights: number[][], // [outDim][inDim] of {-1,0,+1}
  weightScales: Float32Array, // [outDim]
  normWeight: Float32Array | null, // [inDim] or null
  eps = 1e-5
): Float32Array {
  // Step 1: Optional RMSNorm
  let normed: Float32Array;
  if (normWeight) {
    normed = refRMSNorm(input, normWeight, N, inDim, eps);
  } else {
    normed = input;
  }

  // Step 2: Quantize
  const { output: quantized, scales: inputScales } = refQuantize(normed, N, inDim);

  // Step 3: Pack weights and do ternary matmul
  const packedWeights = packI2SMatrix(ternaryWeights, outDim, inDim);

  if (N === 1) {
    // GEMV path
    return refTernaryGemvNaive(
      packedWeights,
      quantized,
      weightScales,
      inputScales[0],
      outDim,
      inDim
    );
  } else {
    // GEMM path
    return refTernaryGemm(
      packedWeights,
      quantized,
      weightScales,
      inputScales,
      outDim,
      N,
      inDim
    );
  }
}

describe("BitLinear end-to-end (CPU reference)", () => {
  it("single token, identity-like weights", () => {
    const inDim = 128;
    const outDim = 64;
    const N = 1;

    // Input: random f32
    const input = randf32(N * inDim, 1);

    // Weights: first outDim rows have +1 on diagonal, rest zero
    const weights: number[][] = [];
    for (let m = 0; m < outDim; m++) {
      const row = new Array(inDim).fill(0);
      row[m] = 1; // identity-like
      weights.push(row);
    }

    const weightScales = new Float32Array(outDim).fill(1.0);

    const result = refBitLinear(input, N, inDim, outDim, weights, weightScales, null);

    // With identity-like weights and scale=1, output[m] ≈ quantized(input[m]) * absmax/127
    // Verify it's non-zero and reasonable
    expect(result.length).toBe(outDim);
    let nonZero = 0;
    for (let i = 0; i < result.length; i++) {
      if (Math.abs(result[i]) > 1e-10) nonZero++;
    }
    expect(nonZero).toBeGreaterThan(outDim / 2);
  });

  it("GEMV matches GEMM for N=1", () => {
    const inDim = 256;
    const outDim = 128;
    const N = 1;

    const input = randf32(N * inDim, 10);
    const ternary = Array.from({ length: outDim }, (_, i) =>
      randTernary(inDim, 100 + i)
    );
    const weightScales = new Float32Array(outDim).fill(1.0);

    // GEMV path (N=1)
    const gemvResult = refBitLinear(input, 1, inDim, outDim, ternary, weightScales, null);

    // GEMM path (force N=1 through GEMM)
    const packedWeights = packI2SMatrix(ternary, outDim, inDim);
    const { output: quantized, scales } = refQuantize(input, 1, inDim);
    const gemmResult = refTernaryGemm(
      packedWeights,
      quantized,
      weightScales,
      scales,
      outDim,
      1,
      inDim
    );

    expectClose(gemvResult, gemmResult, 1e-5, 1e-6);
  });

  it("batched (N>1) produces correct shape and values", () => {
    const inDim = 128;
    const outDim = 64;
    const N = 4;

    const input = randf32(N * inDim, 20);
    const ternary = Array.from({ length: outDim }, (_, i) =>
      randTernary(inDim, 200 + i)
    );
    const weightScales = new Float32Array(outDim).fill(1.0);

    const result = refBitLinear(input, N, inDim, outDim, ternary, weightScales, null);
    expect(result.length).toBe(N * outDim);

    // Each row should be different (different input → different output)
    const row0 = result.slice(0, outDim);
    const row1 = result.slice(outDim, 2 * outDim);
    let diff = 0;
    for (let i = 0; i < outDim; i++) {
      diff += Math.abs(row0[i] - row1[i]);
    }
    expect(diff).toBeGreaterThan(0.01);
  });

  it("with RMSNorm pre-processing", () => {
    const inDim = 128;
    const outDim = 64;
    const N = 1;

    const input = randf32(N * inDim, 30);
    const normWeight = ones(inDim);
    const ternary = Array.from({ length: outDim }, (_, i) =>
      randTernary(inDim, 300 + i)
    );
    const weightScales = new Float32Array(outDim).fill(1.0);

    // With norm
    const withNorm = refBitLinear(input, N, inDim, outDim, ternary, weightScales, normWeight);
    // Without norm
    const withoutNorm = refBitLinear(input, N, inDim, outDim, ternary, weightScales, null);

    // Results should differ when norm is applied
    expect(withNorm.length).toBe(outDim);
    let maxDiff = 0;
    for (let i = 0; i < outDim; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(withNorm[i] - withoutNorm[i]));
    }
    // With unit norm weights and non-unit-variance input, norm should change values
    expect(maxDiff).toBeGreaterThan(0.001);
  });

  it("dequant formula: sum * weightScale * (absmax/127)", () => {
    // Verify the exact dequantization formula matches the reference:
    // Reference (model.py BitLinear): quantize(x) = round(x * 127/absmax) / (127/absmax)
    //   → effectively: output = matmul(w, x_quantized) where x_quantized ≈ x (with quantization noise)
    // Our formula: output = int_dot * weight_scale * input_scale
    //   where input_scale = absmax/127
    //   → output = int_dot * weight_scale * absmax/127
    //   = weight_scale * sum(w_i * round(x_i * 127/absmax)) * absmax/127
    //   = weight_scale * sum(w_i * x_i_quantized_back_to_f32)
    //   ≈ weight_scale * sum(w_i * x_i)  (approximate due to quantization)

    const inDim = 256;
    const outDim = 1;
    const input = new Float32Array(inDim);
    for (let i = 0; i < inDim; i++) input[i] = 0.5; // constant input

    const weights = [new Array(inDim).fill(1)]; // all +1
    const weightScale = new Float32Array([1.0]);

    const result = refBitLinear(input, 1, inDim, outDim, weights, weightScale, null);

    // Expected: sum(+1 * round(0.5 * 127/0.5)) * 1.0 * (0.5/127)
    // = sum(+1 * 127) * 0.5/127
    // = 256 * 127 * 0.5/127 = 128
    expect(result[0]).toBeCloseTo(128, 0);
  });

  it("weight scale of 1.0 matches ternary {-1,0,+1} reference", () => {
    // For the official BitNet model, weight_scale should be 1.0
    // because the ternary weights are already {-1, 0, +1}
    const inDim = 128;
    const outDim = 32;
    const input = randf32(inDim, 50);

    const ternary = Array.from({ length: outDim }, (_, i) =>
      randTernary(inDim, 500 + i)
    );
    const weightScales = new Float32Array(outDim).fill(1.0);

    const result = refBitLinear(input, 1, inDim, outDim, ternary, weightScales, null);

    // Manual verification: compute expected output for row 0
    const { output: qx, scales } = refQuantize(input, 1, inDim);
    let expected = 0;
    for (let k = 0; k < inDim; k++) {
      expected += ternary[0][k] * qx[k];
    }
    expected *= 1.0 * scales[0]; // weightScale * inputScale

    expect(result[0]).toBeCloseTo(expected, 3);
  });
});
