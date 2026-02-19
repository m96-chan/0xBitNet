import { describe, it } from "vitest";
import { refEmbedding } from "./helpers/reference.js";
import { expectClose, randf32 } from "./helpers/tensor-utils.js";

describe("embedding", () => {
  it("looks up correct rows", () => {
    const V = 10;
    const D = 4;
    const embedTable = randf32(V * D, 1);
    const tokenIds = new Uint32Array([0, 3, 7]);
    const output = refEmbedding(tokenIds, embedTable, V, D);

    // Token 0 → row 0
    expectClose(output.slice(0, D), embedTable.slice(0, D));
    // Token 3 → row 3
    expectClose(output.slice(D, 2 * D), embedTable.slice(3 * D, 4 * D));
    // Token 7 → row 7
    expectClose(output.slice(2 * D, 3 * D), embedTable.slice(7 * D, 8 * D));
  });

  it("out-of-vocab token yields zeros", () => {
    const V = 5;
    const D = 4;
    const embedTable = randf32(V * D, 2);
    const tokenIds = new Uint32Array([999]);
    const output = refEmbedding(tokenIds, embedTable, V, D);

    for (let i = 0; i < D; i++) {
      expectClose([output[i]], [0]);
    }
  });

  it("handles multiple tokens including out-of-vocab", () => {
    const V = 3;
    const D = 2;
    const embedTable = new Float32Array([1, 2, 3, 4, 5, 6]);
    const tokenIds = new Uint32Array([0, 100, 2]);
    const output = refEmbedding(tokenIds, embedTable, V, D);

    expectClose(output, new Float32Array([1, 2, 0, 0, 5, 6]));
  });
});
