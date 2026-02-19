import { describe, it, expect } from "vitest";
import { packI2SMatrix } from "./helpers/i2s-pack.js";
import {
  refTernaryGemvNaive,
  refTernaryGemvShaderLogic,
} from "./helpers/reference.js";
import { expectClose, randTernary, randInt8 } from "./helpers/tensor-utils.js";

describe("ternary GEMV", () => {
  const M = 8;
  const K = 256;

  function setup(seed = 42) {
    const weightRows: number[][] = [];
    for (let m = 0; m < M; m++) {
      weightRows.push(randTernary(K, seed + m));
    }
    const packed = packI2SMatrix(weightRows, M, K);
    const input = randInt8(K, seed + 100);
    const weightScales = new Float32Array(M);
    for (let m = 0; m < M; m++) {
      weightScales[m] = 0.01 + 0.001 * m;
    }
    const inputScale = 0.05;
    return { weightRows, packed, input, weightScales, inputScale };
  }

  it("naive vs shader-logic extraction agree", () => {
    const { packed, input, weightScales, inputScale } = setup();
    const naive = refTernaryGemvNaive(packed, input, weightScales, inputScale, M, K);
    const shaderLogic = refTernaryGemvShaderLogic(packed, input, weightScales, inputScale, M, K);
    expectClose(naive, shaderLogic, 1e-5);
  });

  it("known small example: K=128", () => {
    const smallM = 2;
    const smallK = 128;
    // Weight row 0: all +1
    // Weight row 1: all -1
    const rows = [
      new Array(smallK).fill(1),
      new Array(smallK).fill(-1),
    ];
    const packed = packI2SMatrix(rows, smallM, smallK);
    const input = new Int32Array(smallK);
    for (let i = 0; i < smallK; i++) input[i] = 1;
    const weightScales = new Float32Array([1.0, 1.0]);
    const inputScale = 1.0;

    const naive = refTernaryGemvNaive(packed, input, weightScales, inputScale, smallM, smallK);
    expect(naive[0]).toBe(128); // all +1 dot all 1 = 128
    expect(naive[1]).toBe(-128); // all -1 dot all 1 = -128
  });

  it("zero weights → zero output", () => {
    const rows = Array.from({ length: M }, () => new Array(K).fill(0));
    const packed = packI2SMatrix(rows, M, K);
    const input = randInt8(K, 1);
    const weightScales = new Float32Array(M).fill(1.0);
    const naive = refTernaryGemvNaive(packed, input, weightScales, 1.0, M, K);
    for (let m = 0; m < M; m++) {
      expect(naive[m]).toBe(0);
    }
  });

  it("scale applied correctly", () => {
    const { packed, input, inputScale } = setup();
    const scalesA = new Float32Array(M).fill(1.0);
    const scalesB = new Float32Array(M).fill(2.0);

    const outA = refTernaryGemvNaive(packed, input, scalesA, inputScale, M, K);
    const outB = refTernaryGemvNaive(packed, input, scalesB, inputScale, M, K);

    for (let m = 0; m < M; m++) {
      expectClose([outB[m]], [outA[m] * 2], 1e-5);
    }
  });

  it("K=384 (3 blocks)", () => {
    const bigK = 384;
    const bigM = 4;
    const rows: number[][] = [];
    for (let m = 0; m < bigM; m++) {
      rows.push(randTernary(bigK, 50 + m));
    }
    const packed = packI2SMatrix(rows, bigM, bigK);
    const input = randInt8(bigK, 200);
    const weightScales = new Float32Array(bigM).fill(0.01);
    const inputScale = 0.03;

    const naive = refTernaryGemvNaive(packed, input, weightScales, inputScale, bigM, bigK);
    const shaderLogic = refTernaryGemvShaderLogic(packed, input, weightScales, inputScale, bigM, bigK);
    expectClose(naive, shaderLogic, 1e-5);
  });
});
