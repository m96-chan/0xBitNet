import { describe, it } from "vitest";
import { refActivation } from "./helpers/reference.js";
import { expectClose } from "./helpers/tensor-utils.js";

describe("activation", () => {
  describe("ReLU² (type=0)", () => {
    it("positive input → squared", () => {
      const input = new Float32Array([2, 3, 0.5]);
      const result = refActivation(input, 0);
      expectClose(result, new Float32Array([4, 9, 0.25]));
    });

    it("negative input → 0", () => {
      const input = new Float32Array([-1, -5, -0.1]);
      const result = refActivation(input, 0);
      expectClose(result, new Float32Array([0, 0, 0]));
    });

    it("zero → 0", () => {
      const input = new Float32Array([0]);
      const result = refActivation(input, 0);
      expectClose(result, new Float32Array([0]));
    });

    it("mixed values", () => {
      const input = new Float32Array([-2, 0, 3, -1, 4]);
      const result = refActivation(input, 0);
      expectClose(result, new Float32Array([0, 0, 9, 0, 16]));
    });
  });

  describe("SiLU (type=1)", () => {
    it("positive input", () => {
      const input = new Float32Array([1]);
      const result = refActivation(input, 1);
      // SiLU(1) = 1 / (1 + exp(-1)) ≈ 0.7311
      expectClose(result, new Float32Array([1 / (1 + Math.exp(-1))]), 1e-5);
    });

    it("zero → 0", () => {
      const input = new Float32Array([0]);
      const result = refActivation(input, 1);
      expectClose(result, new Float32Array([0]));
    });

    it("negative input", () => {
      const input = new Float32Array([-2]);
      const result = refActivation(input, 1);
      expectClose(result, new Float32Array([-2 / (1 + Math.exp(2))]), 1e-5);
    });

    it("large positive → ≈ x", () => {
      const input = new Float32Array([10]);
      const result = refActivation(input, 1);
      // sigmoid(10) ≈ 1, so SiLU(10) ≈ 10
      expectClose(result, new Float32Array([10]), 1e-3);
    });
  });
});
