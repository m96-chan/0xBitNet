import { describe, it } from "vitest";
import { refRMSNorm } from "./helpers/reference.js";
import { expectClose, randf32, ones } from "./helpers/tensor-utils.js";

describe("rmsnorm", () => {
  it("unit weight: normalizes to unit RMS", () => {
    const N = 1;
    const D = 4;
    const input = new Float32Array([1, 2, 3, 4]);
    const weight = ones(D);
    const eps = 1e-5;
    const output = refRMSNorm(input, weight, N, D, eps);

    // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5)
    const rms = Math.sqrt(30 / 4 + eps);
    const expected = new Float32Array([1 / rms, 2 / rms, 3 / rms, 4 / rms]);
    expectClose(output, expected, 1e-5);
  });

  it("weight scaling", () => {
    const N = 1;
    const D = 4;
    const input = new Float32Array([1, 2, 3, 4]);
    const weight = new Float32Array([2, 2, 2, 2]);
    const eps = 1e-5;
    const output = refRMSNorm(input, weight, N, D, eps);

    const unitOutput = refRMSNorm(input, ones(D), N, D, eps);
    const expected = new Float32Array(unitOutput.map((v) => v * 2));
    expectClose(output, expected, 1e-5);
  });

  it("multiple rows", () => {
    const N = 2;
    const D = 4;
    const input = new Float32Array([1, 2, 3, 4, -1, -2, -3, -4]);
    const weight = ones(D);
    const eps = 1e-5;
    const output = refRMSNorm(input, weight, N, D, eps);

    // Both rows have same magnitudes → second row is negative of first
    expectClose(
      output.slice(D, 2 * D),
      output.slice(0, D).map((v) => -v),
      1e-5
    );
  });

  it("eps prevents division by zero for zero input", () => {
    const N = 1;
    const D = 4;
    const input = new Float32Array([0, 0, 0, 0]);
    const weight = ones(D);
    const eps = 1e-5;
    const output = refRMSNorm(input, weight, N, D, eps);

    // All outputs should be 0 (0 * rsqrt(eps) * 1)
    expectClose(output, new Float32Array([0, 0, 0, 0]));
  });

  it("random data preserves shape", () => {
    const N = 8;
    const D = 512;
    const input = randf32(N * D, 42);
    const weight = randf32(D, 43);
    const output = refRMSNorm(input, weight, N, D, 1e-5);
    expectClose([output.length], [N * D]);
  });
});
