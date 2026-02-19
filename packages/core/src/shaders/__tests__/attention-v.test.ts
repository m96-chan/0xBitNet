import { describe, it, expect } from "vitest";
import { refAttentionV } from "./helpers/reference.js";
import { expectClose, randf32 } from "./helpers/tensor-utils.js";

describe("attention V (attn_weights @ V)", () => {
  it("single head, single position", () => {
    const N = 1;
    const S = 2;
    const numHeads = 1;
    const numKVHeads = 1;
    const headDim = 3;

    // attn weights: [1, numHeads=1, N=1, S=2]
    const attn = new Float32Array([0.7, 0.3]);
    // V: [S=2, numKVHeads=1, headDim=3]
    const V = new Float32Array([1, 2, 3, 4, 5, 6]);

    const output = refAttentionV(attn, V, N, S, numHeads, numKVHeads, headDim);

    // output[0] = 0.7*1 + 0.3*4 = 1.9
    // output[1] = 0.7*2 + 0.3*5 = 2.9
    // output[2] = 0.7*3 + 0.3*6 = 3.9
    expectClose(output, new Float32Array([1.9, 2.9, 3.9]), 1e-5);
  });

  it("output layout: [N, num_heads, head_dim]", () => {
    const N = 2;
    const S = 3;
    const numHeads = 2;
    const numKVHeads = 2;
    const headDim = 2;

    const attn = randf32(numHeads * N * S, 1);
    const V = randf32(S * numKVHeads * headDim, 2);

    const output = refAttentionV(attn, V, N, S, numHeads, numKVHeads, headDim);

    expect(output.length).toBe(N * numHeads * headDim);
  });

  it("GQA: heads sharing KV heads", () => {
    const N = 1;
    const S = 1;
    const numHeads = 4;
    const numKVHeads = 2;
    const headDim = 2;

    // All attention weights = 1 (single S position)
    const attn = new Float32Array(numHeads * N * S).fill(1.0);
    // V: [1, 2, headDim=2]
    const V = new Float32Array([10, 20, 30, 40]);

    const output = refAttentionV(attn, V, N, S, numHeads, numKVHeads, headDim);

    // Heads 0,1 → KV head 0 → V=[10, 20]
    // Heads 2,3 → KV head 1 → V=[30, 40]
    expectClose(output.slice(0, 2), new Float32Array([10, 20]));
    expectClose(output.slice(2, 4), new Float32Array([10, 20]));
    expectClose(output.slice(4, 6), new Float32Array([30, 40]));
    expectClose(output.slice(6, 8), new Float32Array([30, 40]));
  });

  it("uniform attention → mean of V", () => {
    const N = 1;
    const S = 4;
    const numHeads = 1;
    const numKVHeads = 1;
    const headDim = 2;

    const attn = new Float32Array([0.25, 0.25, 0.25, 0.25]);
    const V = new Float32Array([
      1, 0,  // s=0
      0, 1,  // s=1
      1, 0,  // s=2
      0, 1,  // s=3
    ]);

    const output = refAttentionV(attn, V, N, S, numHeads, numKVHeads, headDim);
    expectClose(output, new Float32Array([0.5, 0.5]), 1e-5);
  });

  it("concentrated attention picks single V row", () => {
    const N = 1;
    const S = 3;
    const numHeads = 1;
    const numKVHeads = 1;
    const headDim = 2;

    const attn = new Float32Array([0, 1, 0]); // attend to s=1 only
    const V = new Float32Array([10, 20, 30, 40, 50, 60]);

    const output = refAttentionV(attn, V, N, S, numHeads, numKVHeads, headDim);
    expectClose(output, new Float32Array([30, 40]), 1e-5);
  });
});
