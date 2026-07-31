import { beforeAll, afterAll, describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createRunner, params, type Runner } from "../helpers/gpu.js";
import {
  refActivation,
  refDequantize,
  refElementwise,
  refQuantize,
  refRMSNorm,
  refRoPE,
  refSoftmax,
} from "../helpers/reference.js";

const shader = (name: string) =>
  readFileSync(fileURLToPath(new URL(`../../${name}.wgsl`, import.meta.url)), "utf8");

let runner: Runner | null = null;
beforeAll(async () => {
  runner = await createRunner();
}, 120_000);
afterAll(async () => {
  await runner?.close();
});

/**
 * Skips rather than fails without a GPU. CI runs on ubuntu-latest and has none,
 * and a suite that cannot run there should not be the one gating pull requests.
 */
const gpu = (name: string, body: (r: Runner) => Promise<void>) =>
  it(name, async () => {
    if (!runner) return;
    await body(runner);
  }, 60_000);

/**
 * Agreement, not equality: the reference runs in f64 and the shader in f32.
 *
 * An element passes on either measure. Relative alone is useless where the
 * result nears zero through cancellation, and absolute alone is useless where
 * the values are large.
 */
const agree = (
  got: ArrayLike<number>,
  want: ArrayLike<number>,
  { rel = 1e-5, abs = 1e-6 }: { rel?: number; abs?: number } = {},
) => {
  let worstRel = 0;
  for (let i = 0; i < want.length; i += 1) {
    const diff = Math.abs(got[i]! - want[i]!);
    if (diff <= abs) continue;
    worstRel = Math.max(worstRel, diff / Math.max(1e-6, Math.abs(want[i]!)));
  }
  expect(worstRel).toBeLessThan(rel);
};

const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

describe("rmsnorm.wgsl", () => {
  // D deliberately spans the 256-wide workgroup: below it, exactly on it, and
  // not a multiple of it — the strided loop and the reduction both depend on
  // which of those it is.
  for (const [N, D] of [[2, 8], [1, 256], [3, 300], [2, 2560]] as const) {
    gpu(`matches the reference at N=${N} D=${D}`, async (r) => {
      const input = wave(N * D);
      const weight = Float32Array.from({ length: D }, (_, i) => 0.5 + Math.cos(i * 0.11) * 0.4);
      const eps = 1e-5;
      const [out] = await r.run({
        code: shader("rmsnorm"),
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "out", type: "f32", length: N * D },
          { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", eps]]) },
        ],
        workgroups: [N],
      });
      agree(out!, refRMSNorm(input, weight, N, D, eps));
    });
  }

  gpu("keeps eps meaningful on an all-but-zero row", async (r) => {
    // Without this, dropping eps entirely still passes: on ordinary input
    // sumSq/D dwarfs it. Here it is the only thing standing between the
    // reciprocal square root and a division by zero.
    const D = 8;
    const input = new Float32Array(D);
    const weight = new Float32Array(D).fill(1);
    const eps = 1e-5;
    const [out] = await r.run({
      code: shader("rmsnorm"),
      bindings: [
        { kind: "storage", data: input },
        { kind: "storage", data: weight },
        { kind: "out", type: "f32", length: D },
        { kind: "uniform", data: params([["u32", 1], ["u32", D], ["f32", eps]]) },
      ],
      workgroups: [1],
    });
    expect([...out!].every(Number.isFinite)).toBe(true);
    agree(out!, refRMSNorm(input, weight, 1, D, eps));
  });
});

describe("activation.wgsl", () => {
  for (const [label, type] of [["ReLU²", 0], ["SiLU", 1]] as const) {
    gpu(`matches the reference for ${label}`, async (r) => {
      const input = Float32Array.from([-4, -2, -0.5, 0, 0.5, 2, 4, 10, -10, ...wave(64)]);
      const [out] = await r.run({
        code: shader("activation"),
        bindings: [
          { kind: "storage", data: input },
          { kind: "out", type: "f32", length: input.length },
          { kind: "uniform", data: params([["u32", input.length], ["u32", type]]) },
        ],
        workgroups: [Math.ceil(input.length / 256)],
      });
      agree(out!, refActivation(input, type));
    });
  }
});

describe("elementwise.wgsl", () => {
  for (const [label, op] of [["add", 0], ["multiply", 1]] as const) {
    gpu(`matches the reference for ${label}`, async (r) => {
      const a = wave(300);
      const b = wave(300, 0.19);
      const [out] = await r.run({
        code: shader("elementwise"),
        bindings: [
          { kind: "storage", data: a },
          { kind: "storage", data: b },
          { kind: "out", type: "f32", length: a.length },
          { kind: "uniform", data: params([["u32", a.length], ["u32", op]]) },
        ],
        workgroups: [Math.ceil(a.length / 256)],
      });
      agree(out!, refElementwise(a, b, op));
    });
  }
});

describe("softmax.wgsl", () => {
  for (const [N, D] of [[1, 16], [2, 256], [3, 500]] as const) {
    gpu(`matches the reference at N=${N} D=${D}`, async (r) => {
      const input = wave(N * D, 0.23);
      const [out] = await r.run({
        code: shader("softmax"),
        bindings: [
          { kind: "storage", data: input },
          { kind: "out", type: "f32", length: N * D },
          { kind: "uniform", data: params([["u32", N], ["u32", D]]) },
        ],
        workgroups: [N],
      });
      agree(out!, refSoftmax(input, N, D));
    });
  }

  gpu("does not overflow on large logits", async (r) => {
    // The max-subtraction is the whole reason softmax is written the way it is.
    const D = 8;
    const input = Float32Array.from([100, 200, 300, 400, 500, 600, 700, 800]);
    const [out] = await r.run({
      code: shader("softmax"),
      bindings: [
        { kind: "storage", data: input },
        { kind: "out", type: "f32", length: D },
        { kind: "uniform", data: params([["u32", 1], ["u32", D]]) },
      ],
      workgroups: [1],
    });
    expect([...out!].every(Number.isFinite)).toBe(true);
    agree(out!, refSoftmax(input, 1, D));
  });
});

describe("rope.wgsl", () => {
  for (const posOffset of [0, 7]) {
    gpu(`matches the reference at pos_offset=${posOffset}`, async (r) => {
      const [N, heads, headDim, theta] = [3, 4, 16, 10000];
      const input = wave(N * heads * headDim, 0.29);
      const [out] = await r.run({
        code: shader("rope"),
        bindings: [
          { kind: "storage", data: input },
          { kind: "out", type: "f32", length: input.length },
          {
            kind: "uniform",
            data: params([
              ["u32", N], ["u32", heads], ["u32", headDim], ["u32", posOffset], ["f32", theta],
            ]),
          },
        ],
        workgroups: [Math.ceil(input.length / 256)],
      });
      // Loosened on measurement, not on principle. This GPU's sin and cos carry
      // up to 1.86e-4 of absolute error — three orders of magnitude worse than
      // f32 epsilon (1.2e-7) — and RoPE calls both per element. The kernel and
      // the reference compute the same expression; `pow` was checked separately
      // and agrees to 2.8e-7, so the transcendentals are the whole difference.
      //
      // Nothing tighter is achievable here, and a test that demanded it would
      // be reporting the hardware rather than the shader.
      agree(out!, refRoPE(input, N, heads, headDim, posOffset, theta), { abs: 1e-3 });
    });
  }
});

describe("quantize.wgsl", () => {
  for (const [N, D] of [[1, 64], [2, 300]] as const) {
    gpu(`matches the reference at N=${N} D=${D}`, async (r) => {
      const input = wave(N * D, 0.41);
      const [out, scales] = await r.run({
        code: shader("quantize"),
        bindings: [
          { kind: "storage", data: input },
          { kind: "out", type: "i32", length: N * D },
          { kind: "out", type: "f32", length: N },
          { kind: "uniform", data: params([["u32", N], ["u32", D]]) },
        ],
        workgroups: [N],
      });
      const want = refQuantize(input, N, D);
      // Integers: rounding must agree exactly, not approximately.
      expect([...out!]).toEqual([...want.output]);
      agree(scales!, want.scales);
    });
  }
});

describe("dequantize.wgsl", () => {
  gpu("matches the reference", async (r) => {
    const input = Int32Array.from({ length: 300 }, (_, i) => ((i * 37) % 255) - 127);
    const [weightScale, inputScale] = [0.0125, 0.031];
    const [out] = await r.run({
      code: shader("dequantize"),
      bindings: [
        { kind: "storage", data: input },
        { kind: "out", type: "f32", length: input.length },
        { kind: "uniform", data: params([["f32", weightScale]]) },
        { kind: "uniform", data: params([["f32", inputScale]]) },
        { kind: "uniform", data: params([["u32", input.length]]) },
      ],
      workgroups: [Math.ceil(input.length / 256)],
    });
    agree(out!, refDequantize(input, weightScale, inputScale));
  });
});
