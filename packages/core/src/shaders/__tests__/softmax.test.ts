import { describe, it, expect } from "vitest";
import { refSoftmax } from "./helpers/reference.js";
import { expectClose, randf32 } from "./helpers/tensor-utils.js";

describe("softmax", () => {
  it("uniform input → uniform output", () => {
    const N = 1;
    const D = 4;
    const input = new Float32Array([1, 1, 1, 1]);
    const output = refSoftmax(input, N, D);
    expectClose(output, new Float32Array([0.25, 0.25, 0.25, 0.25]));
  });

  it("row sums to 1", () => {
    const N = 3;
    const D = 5;
    const input = randf32(N * D, 1);
    const output = refSoftmax(input, N, D);

    for (let row = 0; row < N; row++) {
      let sum = 0;
      for (let d = 0; d < D; d++) {
        sum += output[row * D + d];
      }
      expect(Math.abs(sum - 1)).toBeLessThan(1e-5);
    }
  });

  it("all values non-negative", () => {
    const N = 2;
    const D = 10;
    const input = randf32(N * D, 2);
    const output = refSoftmax(input, N, D);

    for (let i = 0; i < output.length; i++) {
      expect(output[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it("largest input gets largest probability", () => {
    const input = new Float32Array([1, 5, 2, 3]);
    const output = refSoftmax(input, 1, 4);
    // Index 1 (value 5) should be largest
    expect(output[1]).toBeGreaterThan(output[0]);
    expect(output[1]).toBeGreaterThan(output[2]);
    expect(output[1]).toBeGreaterThan(output[3]);
  });

  it("numerically stable with large values", () => {
    const input = new Float32Array([1000, 1001, 1002]);
    const output = refSoftmax(input, 1, 3);

    let sum = 0;
    for (let i = 0; i < 3; i++) {
      expect(isFinite(output[i])).toBe(true);
      sum += output[i];
    }
    expect(Math.abs(sum - 1)).toBeLessThan(1e-5);
  });

  it("multiple independent rows", () => {
    const input = new Float32Array([
      0, 0, 100, 0, // row 0: index 2 dominates
      100, 0, 0, 0, // row 1: index 0 dominates
    ]);
    const output = refSoftmax(input, 2, 4);

    expect(output[2]).toBeGreaterThan(0.99); // row 0, col 2
    expect(output[4]).toBeGreaterThan(0.99); // row 1, col 0
  });
});
