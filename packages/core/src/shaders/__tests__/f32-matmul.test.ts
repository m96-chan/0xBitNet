import { describe, it, expect } from "vitest";
import { refF32Matmul } from "./helpers/reference.js";
import { expectClose, randf32 } from "./helpers/tensor-utils.js";

describe("f32 matmul (tied-embedding LM head)", () => {
  it("simple dot products", () => {
    const N = 1;
    const V = 2;
    const D = 3;

    const hidden = new Float32Array([1, 2, 3]);
    const embed = new Float32Array([
      1, 0, 0, // vocab 0
      0, 1, 0, // vocab 1
    ]);

    const output = refF32Matmul(hidden, embed, N, V, D);

    expect(output[0]).toBeCloseTo(1.0); // dot([1,2,3], [1,0,0]) = 1
    expect(output[1]).toBeCloseTo(2.0); // dot([1,2,3], [0,1,0]) = 2
  });

  it("output layout [N, V]", () => {
    const N = 2;
    const V = 3;
    const D = 4;

    const hidden = randf32(N * D, 1);
    const embed = randf32(V * D, 2);

    const output = refF32Matmul(hidden, embed, N, V, D);
    expect(output.length).toBe(N * V);
  });

  it("multiple tokens", () => {
    const N = 3;
    const V = 2;
    const D = 2;

    const hidden = new Float32Array([1, 0, 0, 1, 1, 1]);
    const embed = new Float32Array([1, 0, 0, 1]);

    const output = refF32Matmul(hidden, embed, N, V, D);

    // Token 0: [1,0] · [1,0]=1, [1,0] · [0,1]=0
    expect(output[0]).toBeCloseTo(1);
    expect(output[1]).toBeCloseTo(0);
    // Token 1: [0,1] · [1,0]=0, [0,1] · [0,1]=1
    expect(output[2]).toBeCloseTo(0);
    expect(output[3]).toBeCloseTo(1);
    // Token 2: [1,1] · [1,0]=1, [1,1] · [0,1]=1
    expect(output[4]).toBeCloseTo(1);
    expect(output[5]).toBeCloseTo(1);
  });

  it("zero hidden → zero logits", () => {
    const N = 1;
    const V = 5;
    const D = 4;

    const hidden = new Float32Array(D); // all zeros
    const embed = randf32(V * D, 3);

    const output = refF32Matmul(hidden, embed, N, V, D);
    for (let i = 0; i < V; i++) {
      expect(output[i]).toBeCloseTo(0);
    }
  });

  it("consistency with manual dot product", () => {
    const N = 2;
    const V = 3;
    const D = 8;

    const hidden = randf32(N * D, 10);
    const embed = randf32(V * D, 20);

    const output = refF32Matmul(hidden, embed, N, V, D);

    for (let n = 0; n < N; n++) {
      for (let v = 0; v < V; v++) {
        let expected = 0;
        for (let d = 0; d < D; d++) {
          expected += hidden[n * D + d] * embed[v * D + d];
        }
        expectClose([output[n * V + v]], [expected], 1e-5);
      }
    }
  });
});
