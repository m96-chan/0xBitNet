import { describe, it, expect } from "vitest";
import { packI2S, unpackI2S, packI2SMatrix, unpackI2SMatrix } from "./helpers/i2s-pack.js";
import { randTernary } from "./helpers/tensor-utils.js";

describe("I2S pack/unpack", () => {
  it("round-trips a small known pattern", () => {
    // 128 elements = exactly 1 block
    const values = new Array(128).fill(0);
    values[0] = -1; // group0, gp=0
    values[1] = 1;  // group0, gp=1
    values[32] = 1; // group1, gp=0
    values[64] = -1; // group2, gp=0
    values[96] = 1; // group3, gp=0

    const packed = packI2S(values, 128);
    const unpacked = unpackI2S(packed, 128);

    expect(unpacked).toEqual(values);
  });

  it("round-trips random K=128", () => {
    const values = randTernary(128, 1);
    const packed = packI2S(values, 128);
    const unpacked = unpackI2S(packed, 128);
    expect(unpacked).toEqual(values);
  });

  it("round-trips random K=256 (2 blocks)", () => {
    const values = randTernary(256, 2);
    const packed = packI2S(values, 256);
    const unpacked = unpackI2S(packed, 256);
    expect(unpacked).toEqual(values);
  });

  it("round-trips K=384 (3 blocks)", () => {
    const values = randTernary(384, 3);
    const packed = packI2S(values, 384);
    const unpacked = unpackI2S(packed, 384);
    expect(unpacked).toEqual(values);
  });

  it("round-trips non-128-aligned K=200", () => {
    const values = randTernary(200, 4);
    const packed = packI2S(values, 200);
    const unpacked = unpackI2S(packed, 200);
    // First 200 should match; rest is padding
    expect(unpacked.slice(0, 200)).toEqual(values);
  });

  it("round-trips a weight matrix", () => {
    const M = 4;
    const K = 256;
    const rows: number[][] = [];
    for (let m = 0; m < M; m++) {
      rows.push(randTernary(K, 10 + m));
    }
    const packed = packI2SMatrix(rows, M, K);
    const unpacked = unpackI2SMatrix(packed, M, K);
    for (let m = 0; m < M; m++) {
      expect(unpacked[m]).toEqual(rows[m]);
    }
  });

  it("known byte layout: all -1 encodes to 0x00", () => {
    // -1 → code 0, so byte = (0<<6)|(0<<4)|(0<<2)|0 = 0x00
    const values = new Array(128).fill(-1);
    const packed = packI2S(values, 128);
    // Check first u32 (4 bytes, all 0x00)
    expect(packed[0]).toBe(0x00000000);
  });

  it("known byte layout: all +1 encodes to 0xAA", () => {
    // +1 → code 2, so byte = (2<<6)|(2<<4)|(2<<2)|2 = 0b10101010 = 0xAA
    const values = new Array(128).fill(1);
    const packed = packI2S(values, 128);
    // First u32 in little-endian: [0xAA, 0xAA, 0xAA, 0xAA] = 0xAAAAAAAA
    expect(packed[0]).toBe(0xAAAAAAAA);
  });

  it("known byte layout: all 0 encodes to 0x55", () => {
    // 0 → code 1, so byte = (1<<6)|(1<<4)|(1<<2)|1 = 0b01010101 = 0x55
    const values = new Array(128).fill(0);
    const packed = packI2S(values, 128);
    expect(packed[0]).toBe(0x55555555);
  });
});
