import { describe, it, expect } from "vitest";
import { refQuantize } from "./helpers/reference.js";
import { expectClose, randf32 } from "./helpers/tensor-utils.js";

describe("quantize (absmax int8)", () => {
  it("simple known values", () => {
    const N = 1;
    const D = 4;
    const input = new Float32Array([1.0, -1.0, 0.5, 0.0]);
    const { output, scales } = refQuantize(input, N, D);

    // absmax = 1.0, scale = 1/127
    expect(Math.abs(scales[0] - 1 / 127)).toBeLessThan(1e-6);
    expect(output[0]).toBe(127);  // 1.0 * 127 = 127
    expect(output[1]).toBe(-127); // -1.0 * 127 = -127
    expect(output[2]).toBe(64);   // 0.5 * 127 = 63.5 → round to 64
    expect(output[3]).toBe(0);    // 0.0 → 0
  });

  it("scale formula: absmax / 127", () => {
    const N = 1;
    const D = 4;
    const input = new Float32Array([3.0, -5.0, 2.0, 1.0]);
    const { scales } = refQuantize(input, N, D);

    // absmax = 5.0, scale = 5/127
    expectClose([scales[0]], [5 / 127], 1e-6);
  });

  it("clamped to [-127, 127]", () => {
    const N = 1;
    const D = 3;
    const input = new Float32Array([10, -10, 5]);
    const { output } = refQuantize(input, N, D);

    for (let i = 0; i < D; i++) {
      expect(output[i]).toBeGreaterThanOrEqual(-127);
      expect(output[i]).toBeLessThanOrEqual(127);
    }
  });

  it("zero input → zero output, scale = 1", () => {
    const N = 1;
    const D = 4;
    const input = new Float32Array([0, 0, 0, 0]);
    const { output, scales } = refQuantize(input, N, D);

    expect(scales[0]).toBe(1.0);
    for (let i = 0; i < D; i++) {
      expect(output[i]).toBe(0);
    }
  });

  it("per-token independent scales", () => {
    const N = 2;
    const D = 3;
    const input = new Float32Array([
      1, 2, 3,   // absmax = 3, scale = 3/127
      10, 5, 1,  // absmax = 10, scale = 10/127
    ]);
    const { scales } = refQuantize(input, N, D);

    expectClose([scales[0]], [3 / 127], 1e-6);
    expectClose([scales[1]], [10 / 127], 1e-6);
  });

  it("dequantization recovers approximate values", () => {
    const N = 4;
    const D = 128;
    const input = randf32(N * D, 42);
    const { output, scales } = refQuantize(input, N, D);

    // Dequantize: val ≈ output[i] * scale
    for (let row = 0; row < N; row++) {
      for (let d = 0; d < D; d++) {
        const original = input[row * D + d];
        const dequantized = output[row * D + d] * scales[row];
        // int8 quantization error should be small relative to absmax
        expect(Math.abs(original - dequantized)).toBeLessThan(scales[row] * 1.5);
      }
    }
  });
});
