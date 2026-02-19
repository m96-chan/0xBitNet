/**
 * Test utilities for tensor operations.
 */
import { expect } from "vitest";

/**
 * Simple seeded PRNG (mulberry32).
 */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seeded random f32 array in [-1, 1] */
export function randf32(n: number, seed = 42): Float32Array {
  const rng = mulberry32(seed);
  const arr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    arr[i] = rng() * 2 - 1;
  }
  return arr;
}

/** Seeded random ternary array {-1, 0, +1} */
export function randTernary(n: number, seed = 42): number[] {
  const rng = mulberry32(seed);
  const arr: number[] = [];
  for (let i = 0; i < n; i++) {
    const r = rng();
    arr.push(r < 0.333 ? -1 : r < 0.666 ? 0 : 1);
  }
  return arr;
}

/** Seeded random int8 array [-127, 127] stored as i32 */
export function randInt8(n: number, seed = 42): Int32Array {
  const rng = mulberry32(seed);
  const arr = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    arr[i] = Math.floor(rng() * 255) - 127;
  }
  return arr;
}

/** Element-wise approximate equality check */
export function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  rtol = 1e-5,
  atol = 1e-6
): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const tol = atol + rtol * Math.abs(e);
    if (Math.abs(a - e) > tol) {
      throw new Error(
        `Mismatch at index ${i}: actual=${a}, expected=${e}, diff=${Math.abs(a - e)}, tol=${tol}`
      );
    }
  }
}

/** All-zeros f32 array */
export function zeros(n: number): Float32Array {
  return new Float32Array(n);
}

/** All-ones f32 array */
export function ones(n: number): Float32Array {
  const arr = new Float32Array(n);
  arr.fill(1);
  return arr;
}
