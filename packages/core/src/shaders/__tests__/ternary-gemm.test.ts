import { describe, it, expect } from "vitest";
import { packI2SMatrix } from "./helpers/i2s-pack.js";
import {
  refTernaryGemm,
  refTernaryGemvNaive,
} from "./helpers/reference.js";
import { expectClose, randTernary, randInt8 } from "./helpers/tensor-utils.js";

describe("ternary GEMM", () => {
  it("N=1 matches GEMV result", () => {
    const M = 8;
    const K = 256;
    const N = 1;

    const rows: number[][] = [];
    for (let m = 0; m < M; m++) {
      rows.push(randTernary(K, 10 + m));
    }
    const packed = packI2SMatrix(rows, M, K);
    const input = randInt8(K, 100);
    const weightScales = new Float32Array(M);
    for (let m = 0; m < M; m++) weightScales[m] = 0.01 + 0.001 * m;
    const inputScales = new Float32Array([0.05]);

    const gemmOut = refTernaryGemm(packed, input, weightScales, inputScales, M, N, K);
    const gemvOut = refTernaryGemvNaive(packed, input, weightScales, inputScales[0], M, K);

    // GEMM output is [N, M] = [1, M], GEMV output is [M]
    expectClose(gemmOut, gemvOut, 1e-5);
  });

  it("output layout is [N, M]", () => {
    const M = 4;
    const K = 128;
    const N = 3;

    const rows: number[][] = [];
    for (let m = 0; m < M; m++) {
      rows.push(randTernary(K, 20 + m));
    }
    const packed = packI2SMatrix(rows, M, K);
    const input = randInt8(N * K, 200);
    const weightScales = new Float32Array(M).fill(0.01);
    const inputScales = new Float32Array(N);
    for (let n = 0; n < N; n++) inputScales[n] = 0.01 + 0.005 * n;

    const output = refTernaryGemm(packed, input, weightScales, inputScales, M, N, K);

    // Verify dimensions: output should have N*M elements
    expect(output.length).toBe(N * M);

    // Verify each (n, m) independently
    for (let n = 0; n < N; n++) {
      const rowInput = input.slice(n * K, (n + 1) * K);
      const singleOutput = refTernaryGemvNaive(
        packed,
        rowInput,
        weightScales,
        inputScales[n],
        M,
        K
      );
      for (let m = 0; m < M; m++) {
        expectClose([output[n * M + m]], [singleOutput[m]], 1e-5);
      }
    }
  });

  it("per-token input scales", () => {
    const M = 4;
    const K = 128;
    const N = 2;

    const rows: number[][] = [];
    for (let m = 0; m < M; m++) rows.push(randTernary(K, 30 + m));
    const packed = packI2SMatrix(rows, M, K);
    const input = randInt8(N * K, 300);
    const weightScales = new Float32Array(M).fill(1.0);

    const scalesA = new Float32Array([1.0, 1.0]);
    const scalesB = new Float32Array([2.0, 3.0]);

    const outA = refTernaryGemm(packed, input, weightScales, scalesA, M, N, K);
    const outB = refTernaryGemm(packed, input, weightScales, scalesB, M, N, K);

    // Token 0 should be 2x, token 1 should be 3x
    for (let m = 0; m < M; m++) {
      expectClose([outB[0 * M + m]], [outA[0 * M + m] * 2], 1e-5);
      expectClose([outB[1 * M + m]], [outA[1 * M + m] * 3], 1e-5);
    }
  });

  it("known small example: identity-like weights", () => {
    // K=128, M=2, weights: row0 = [+1, +1, ...], row1 = [-1, -1, ...]
    const M = 2;
    const K = 128;
    const N = 2;

    const rows = [
      new Array(K).fill(1),
      new Array(K).fill(-1),
    ];
    const packed = packI2SMatrix(rows, M, K);
    const input = new Int32Array(N * K);
    for (let i = 0; i < N * K; i++) input[i] = 1;
    const weightScales = new Float32Array([1, 1]);
    const inputScales = new Float32Array([1, 1]);

    const output = refTernaryGemm(packed, input, weightScales, inputScales, M, N, K);

    // Each token: dot(all-1-weights, all-1-input) = 128, dot(all-(-1), all-1) = -128
    expect(output[0 * M + 0]).toBe(128);
    expect(output[0 * M + 1]).toBe(-128);
    expect(output[1 * M + 0]).toBe(128);
    expect(output[1 * M + 1]).toBe(-128);
  });

  it("K=384 multi-block", () => {
    const M = 4;
    const K = 384;
    const N = 2;

    const rows: number[][] = [];
    for (let m = 0; m < M; m++) rows.push(randTernary(K, 40 + m));
    const packed = packI2SMatrix(rows, M, K);
    const input = randInt8(N * K, 400);
    const weightScales = new Float32Array(M).fill(0.01);
    const inputScales = new Float32Array(N).fill(0.02);

    const output = refTernaryGemm(packed, input, weightScales, inputScales, M, N, K);

    // Cross-check each token with GEMV
    for (let n = 0; n < N; n++) {
      const rowInput = input.slice(n * K, (n + 1) * K);
      const gemvOut = refTernaryGemvNaive(
        packed,
        rowInput,
        weightScales,
        inputScales[n],
        M,
        K
      );
      for (let m = 0; m < M; m++) {
        expectClose([output[n * M + m]], [gemvOut[m]], 1e-5);
      }
    }
  });
});
