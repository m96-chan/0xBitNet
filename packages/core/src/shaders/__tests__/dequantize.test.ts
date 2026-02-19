import { describe, it } from "vitest";
import { refDequantize } from "./helpers/reference.js";
import { expectClose } from "./helpers/tensor-utils.js";

describe("dequantize", () => {
  it("simple scaling", () => {
    const input = new Int32Array([100, -50, 0, 127]);
    const output = refDequantize(input, 0.5, 0.1);

    expectClose(output, new Float32Array([
      100 * 0.5 * 0.1,
      -50 * 0.5 * 0.1,
      0 * 0.5 * 0.1,
      127 * 0.5 * 0.1,
    ]));
  });

  it("zero weight scale → zero output", () => {
    const input = new Int32Array([100, 50, -127]);
    const output = refDequantize(input, 0, 0.5);
    expectClose(output, new Float32Array([0, 0, 0]));
  });

  it("zero input scale → zero output", () => {
    const input = new Int32Array([100, 50, -127]);
    const output = refDequantize(input, 0.5, 0);
    expectClose(output, new Float32Array([0, 0, 0]));
  });

  it("identity: scales = 1", () => {
    const input = new Int32Array([10, -20, 30]);
    const output = refDequantize(input, 1, 1);
    expectClose(output, new Float32Array([10, -20, 30]));
  });
});
