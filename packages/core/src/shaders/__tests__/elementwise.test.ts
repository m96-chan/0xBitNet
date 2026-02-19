import { describe, it } from "vitest";
import { refElementwise } from "./helpers/reference.js";
import { expectClose, randf32 } from "./helpers/tensor-utils.js";

describe("elementwise", () => {
  it("add: a + b", () => {
    const a = new Float32Array([1, -2, 3, 0.5]);
    const b = new Float32Array([4, 5, -1, 0.5]);
    const result = refElementwise(a, b, 0);
    expectClose(result, new Float32Array([5, 3, 2, 1]));
  });

  it("multiply: a * b", () => {
    const a = new Float32Array([1, -2, 3, 0.5]);
    const b = new Float32Array([4, 5, -1, 0.5]);
    const result = refElementwise(a, b, 1);
    expectClose(result, new Float32Array([4, -10, -3, 0.25]));
  });

  it("add with random data", () => {
    const a = randf32(1024, 1);
    const b = randf32(1024, 2);
    const result = refElementwise(a, b, 0);
    for (let i = 0; i < 1024; i++) {
      expectClose([result[i]], [a[i] + b[i]]);
    }
  });

  it("multiply with random data", () => {
    const a = randf32(1024, 3);
    const b = randf32(1024, 4);
    const result = refElementwise(a, b, 1);
    for (let i = 0; i < 1024; i++) {
      expectClose([result[i]], [a[i] * b[i]]);
    }
  });
});
