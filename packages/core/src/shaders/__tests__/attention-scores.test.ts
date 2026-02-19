import { describe, it, expect } from "vitest";
import { refAttentionScores } from "./helpers/reference.js";
import { expectClose, randf32 } from "./helpers/tensor-utils.js";

const NEG_INF = -3.402823e+38;

function isMasked(val: number): boolean {
  return val < -1e+37;
}

describe("attention scores (Q @ K^T)", () => {
  it("simple dot product", () => {
    const N = 1;
    const S = 1;
    const numHeads = 1;
    const numKVHeads = 1;
    const headDim = 4;
    const scale = 1.0;

    const Q = new Float32Array([1, 0, 1, 0]);
    const K = new Float32Array([1, 0, 1, 0]);

    const scores = refAttentionScores(Q, K, N, S, numHeads, numKVHeads, headDim, scale);
    expect(scores[0]).toBeCloseTo(2.0);
  });

  it("causal mask: future positions masked", () => {
    const N = 3;
    const S = 3;
    const numHeads = 1;
    const numKVHeads = 1;
    const headDim = 2;
    const scale = 1.0;

    const Q = randf32(N * numHeads * headDim, 1);
    const K = randf32(S * numKVHeads * headDim, 2);

    const scores = refAttentionScores(Q, K, N, S, numHeads, numKVHeads, headDim, scale);

    // Row 0 (q_pos=0): only k_pos=0 is visible (k=1,2 are future)
    expect(isMasked(scores[0 * S + 1])).toBe(true);
    expect(isMasked(scores[0 * S + 2])).toBe(true);

    // Row 1 (q_pos=1): k_pos=0,1 visible, k=2 masked
    expect(isMasked(scores[1 * S + 0])).toBe(false);
    expect(isMasked(scores[1 * S + 1])).toBe(false);
    expect(isMasked(scores[1 * S + 2])).toBe(true);

    // Row 2 (q_pos=2): all visible (last row)
    expect(isMasked(scores[2 * S + 0])).toBe(false);
    expect(isMasked(scores[2 * S + 1])).toBe(false);
    expect(isMasked(scores[2 * S + 2])).toBe(false);
  });

  it("causal mask with KV cache (S > N)", () => {
    // Simulating: 4 cached tokens + 2 new tokens
    const N = 2;
    const S = 6;
    const numHeads = 1;
    const numKVHeads = 1;
    const headDim = 2;
    const scale = 0.5;

    const Q = randf32(N * numHeads * headDim, 3);
    const K = randf32(S * numKVHeads * headDim, 4);

    const scores = refAttentionScores(Q, K, N, S, numHeads, numKVHeads, headDim, scale);

    // q_pos=0 maps to absolute pos 4 (S-N=4): can see k_pos 0..4
    expect(isMasked(scores[0 * S + 4])).toBe(false); // pos 4 visible
    expect(isMasked(scores[0 * S + 5])).toBe(true);  // pos 5 masked

    // q_pos=1 maps to absolute pos 5: can see all
    for (let k = 0; k < S; k++) {
      expect(isMasked(scores[1 * S + k])).toBe(false);
    }
  });

  it("GQA: multiple heads share KV heads", () => {
    const N = 1;
    const S = 1;
    const numHeads = 4;
    const numKVHeads = 2; // 2 heads per KV group
    const headDim = 2;
    const scale = 1.0;

    const Q = randf32(N * numHeads * headDim, 5);
    const K = randf32(S * numKVHeads * headDim, 6);

    const scores = refAttentionScores(Q, K, N, S, numHeads, numKVHeads, headDim, scale);

    // Heads 0,1 share KV head 0; heads 2,3 share KV head 1
    // Heads in same group use same K → but different Q → different scores (usually)
    expect(scores.length).toBe(numHeads * N * S);
  });

  it("scale factor applied", () => {
    const N = 1;
    const S = 1;
    const numHeads = 1;
    const numKVHeads = 1;
    const headDim = 4;

    const Q = new Float32Array([1, 1, 1, 1]);
    const K = new Float32Array([1, 1, 1, 1]);

    const s1 = refAttentionScores(Q, K, N, S, numHeads, numKVHeads, headDim, 1.0);
    const s2 = refAttentionScores(Q, K, N, S, numHeads, numKVHeads, headDim, 0.5);

    expect(s1[0]).toBeCloseTo(4.0);
    expect(s2[0]).toBeCloseTo(2.0);
  });
});
