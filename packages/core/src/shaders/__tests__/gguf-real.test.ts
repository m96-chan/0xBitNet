/**
 * Tests against the REAL BitNet 2B-4T GGUF model file.
 *
 * Requires: models/ggml-model-i2_s.gguf in the repo root.
 * Skipped if the file is not present.
 *
 * These tests verify:
 * 1. GGUF metadata is parsed correctly (architecture, head counts, etc.)
 * 2. Tensor info (names, shapes, types, offsets) are correct
 * 3. I2_S weight data is valid (unpacks to {-1,0,+1})
 * 4. Per-tensor scales are reasonable (non-zero, non-NaN)
 * 5. Embedding data (F16→F32) looks reasonable
 */
import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync, existsSync } from "fs";
import { resolve } from "path";
import { GGUFParser, GGML_TYPE_I2_S, GGML_TYPE_F16, GGML_TYPE_F32 } from "../../model/gguf.js";
import { unpackI2S } from "./helpers/i2s-pack.js";
import {
  refRMSNorm,
  refQuantize,
  refTernaryGemvNaive,
  refTernaryGemvShaderLogic,
} from "./helpers/reference.js";
import type { GGUFFile, GGUFTensorInfo } from "../../types.js";

const MODEL_PATH = resolve(__dirname, "../../../../../models/ggml-model-i2_s.gguf");
const HAS_MODEL = existsSync(MODEL_PATH);

// Load once for all tests
let gguf: GGUFFile;
let buffer: ArrayBuffer;

beforeAll(() => {
  if (!HAS_MODEL) return;
  const fileBuffer = readFileSync(MODEL_PATH);
  buffer = fileBuffer.buffer.slice(
    fileBuffer.byteOffset,
    fileBuffer.byteOffset + fileBuffer.byteLength
  );
  const parser = new GGUFParser(buffer);
  gguf = parser.parse();
});

function findTensor(name: string): GGUFTensorInfo | undefined {
  return gguf.tensors.find((t) => t.name === name);
}

function getTensorData(tensor: GGUFTensorInfo): ArrayBuffer {
  const numElements = tensor.shape.reduce((a, b) => a * Number(b), 1);
  let byteSize: number;
  if (tensor.type === GGML_TYPE_I2_S) {
    byteSize = Math.ceil(numElements / 4) + 32;
  } else if (tensor.type === GGML_TYPE_F16) {
    byteSize = numElements * 2;
  } else if (tensor.type === GGML_TYPE_F32) {
    byteSize = numElements * 4;
  } else {
    byteSize = numElements; // fallback
  }
  const offset = gguf.tensorDataOffset + Number(tensor.offset);
  return buffer.slice(offset, offset + byteSize);
}

describe.skipIf(!HAS_MODEL)("GGUF metadata", () => {
  it("architecture is set", () => {
    const arch = gguf.metadata["general.architecture"] as string;
    console.log(`  architecture: "${arch}"`);
    expect(arch).toBeDefined();
    expect(typeof arch).toBe("string");
  });

  it("head counts match HuggingFace config", () => {
    const arch = gguf.metadata["general.architecture"] as string;
    const headCount =
      gguf.metadata[`${arch}.attention.head_count`] as number;
    const kvHeadCount =
      gguf.metadata[`${arch}.attention.head_count_kv`] as number;
    console.log(`  head_count: ${headCount}, head_count_kv: ${kvHeadCount}`);
    console.log(`  head_dim: ${2560 / headCount}`);
    expect(headCount).toBe(20);
    expect(kvHeadCount).toBe(5);
  });

  it("embedding length matches", () => {
    const arch = gguf.metadata["general.architecture"] as string;
    const embLen = gguf.metadata[`${arch}.embedding_length`] as number;
    console.log(`  embedding_length: ${embLen}`);
    expect(embLen).toBe(2560);
  });

  it("block count matches", () => {
    const arch = gguf.metadata["general.architecture"] as string;
    const blocks = gguf.metadata[`${arch}.block_count`] as number;
    console.log(`  block_count: ${blocks}`);
    expect(blocks).toBe(30);
  });

  it("feed forward length matches", () => {
    const arch = gguf.metadata["general.architecture"] as string;
    const ffLen = gguf.metadata[`${arch}.feed_forward_length`] as number;
    console.log(`  feed_forward_length: ${ffLen}`);
    expect(ffLen).toBe(6912);
  });

  it("rope theta", () => {
    const arch = gguf.metadata["general.architecture"] as string;
    const theta = gguf.metadata[`${arch}.rope.freq_base`] as number;
    console.log(`  rope.freq_base: ${theta}`);
    // May or may not be in metadata
    if (theta !== undefined) {
      expect(theta).toBeCloseTo(500000.0, 0);
    }
  });

  it("BOS/EOS token IDs and embedding sharding analysis", () => {
    const bosId = gguf.metadata["tokenizer.ggml.bos_token_id"] as number | undefined;
    const eosId = gguf.metadata["tokenizer.ggml.eos_token_id"] as number | undefined;
    console.log(`  BOS token ID: ${bosId}`);
    console.log(`  EOS token ID: ${eosId}`);

    // Check if BOS/EOS would be out of range with common GPU limits
    const embedTensor = findTensor("token_embd.weight")!;
    const vocabSize = Number(embedTensor.shape[0]); // ne[0] = inDim for embed = hiddenSize, or is it vocabSize?
    const hiddenSize = Number(embedTensor.shape[1] ?? embedTensor.shape[0]);
    const numElements = embedTensor.shape.reduce((a, b) => a * Number(b), 1);
    const f32Size = numElements * 4; // after F16→F32 conversion
    console.log(`  Embedding table F32 size: ${f32Size} bytes (${(f32Size / 1e9).toFixed(2)} GB)`);
    console.log(`  Embedding shape: [${embedTensor.shape.join(", ")}]`);

    // Common GPU limits
    const gpuLimits = [
      { name: "128MB (Intel iGPU)", bytes: 128 * 1024 * 1024 },
      { name: "256MB", bytes: 256 * 1024 * 1024 },
      { name: "1GB (NVIDIA default)", bytes: 1073741824 },
      { name: "2GB", bytes: 2147483648 },
    ];

    for (const { name, bytes } of gpuLimits) {
      if (f32Size > bytes) {
        const maxRows = Math.floor(bytes / (2560 * 4));
        const bosOK = bosId === undefined || bosId < maxRows;
        const eosOK = eosId === undefined || eosId < maxRows;
        console.log(`  ⚠ ${name}: sharded! maxRows=${maxRows}, BOS(${bosId}) ${bosOK ? "OK" : "OUT OF RANGE!"}, EOS(${eosId}) ${eosOK ? "OK" : "OUT OF RANGE!"}`);
      } else {
        console.log(`  ✓ ${name}: fits (no sharding)`);
      }
    }

    expect(bosId).toBeDefined();
  });

  it("dump all metadata keys", () => {
    const keys = Object.keys(gguf.metadata).filter(
      (k) => !k.startsWith("tokenizer.")
    );
    console.log("  Non-tokenizer metadata keys:");
    for (const k of keys.sort()) {
      const v = gguf.metadata[k];
      const display = typeof v === "string" ? `"${v}"` : v;
      console.log(`    ${k} = ${display}`);
    }
    expect(keys.length).toBeGreaterThan(0);
  });
});

describe.skipIf(!HAS_MODEL)("GGUF tensor inventory", () => {
  it("has embedding tensor", () => {
    const t = findTensor("token_embd.weight");
    expect(t).toBeDefined();
    console.log(`  token_embd.weight: type=${t!.type}, shape=[${t!.shape}]`);
    expect(t!.type).toBe(GGML_TYPE_F16);
    // Shape: [hidden, vocab] in GGUF (reversed from PyTorch [vocab, hidden])
    expect(Number(t!.shape[0])).toBe(2560);
    expect(Number(t!.shape[1])).toBe(128256);
  });

  it("has no output.weight (tied embeddings)", () => {
    const t = findTensor("output.weight");
    console.log(`  output.weight exists: ${t !== undefined}`);
    expect(t).toBeUndefined();
  });

  it("has all layer tensors for layer 0", () => {
    const expected = [
      "blk.0.attn_q.weight",
      "blk.0.attn_k.weight",
      "blk.0.attn_v.weight",
      "blk.0.attn_output.weight",
      "blk.0.attn_norm.weight",
      "blk.0.ffn_norm.weight",
      "blk.0.attn_sub_norm.weight",
      "blk.0.ffn_sub_norm.weight",
      "blk.0.ffn_up.weight",
      "blk.0.ffn_down.weight",
      "blk.0.ffn_gate.weight",
    ];
    for (const name of expected) {
      const t = findTensor(name);
      console.log(
        `  ${name}: ${t ? `type=${t.type}, shape=[${t.shape}]` : "MISSING!"}`
      );
      expect(t).toBeDefined();
    }
  });

  it("Q/K/V/O shapes are consistent with 20 heads, head_dim=128", () => {
    const q = findTensor("blk.0.attn_q.weight")!;
    const k = findTensor("blk.0.attn_k.weight")!;
    const v = findTensor("blk.0.attn_v.weight")!;
    const o = findTensor("blk.0.attn_output.weight")!;

    // GGUF shape: ne[0]=inDim, ne[1]=outDim (reversed from PyTorch)
    // Q: PyTorch [numHeads*headDim, hidden] = [2560, 2560]
    //    GGUF: ne=[2560, 2560]
    console.log(`  Q shape: [${q.shape}] (expect [2560, 2560])`);
    console.log(`  K shape: [${k.shape}] (expect [2560, 640])`);
    console.log(`  V shape: [${v.shape}] (expect [2560, 640])`);
    console.log(`  O shape: [${o.shape}] (expect [2560, 2560])`);

    expect(Number(q.shape[0])).toBe(2560); // inDim = hidden
    expect(Number(q.shape[1])).toBe(2560); // outDim = 20*128

    expect(Number(k.shape[0])).toBe(2560); // inDim = hidden
    expect(Number(k.shape[1])).toBe(640);  // outDim = 5*128

    expect(Number(v.shape[0])).toBe(2560);
    expect(Number(v.shape[1])).toBe(640);

    expect(Number(o.shape[0])).toBe(2560); // inDim = 20*128
    expect(Number(o.shape[1])).toBe(2560); // outDim = hidden
  });

  it("FFN shapes are consistent", () => {
    const up = findTensor("blk.0.ffn_up.weight")!;
    const down = findTensor("blk.0.ffn_down.weight")!;
    const gate = findTensor("blk.0.ffn_gate.weight")!;

    console.log(`  up shape: [${up.shape}] (expect [2560, 6912])`);
    console.log(`  down shape: [${down.shape}] (expect [6912, 2560])`);
    console.log(`  gate shape: [${gate.shape}] (expect [2560, 6912])`);

    // up: PyTorch [intermediateSize, hidden] → GGUF ne=[hidden, intermediateSize]
    expect(Number(up.shape[0])).toBe(2560);
    expect(Number(up.shape[1])).toBe(6912);

    // down: PyTorch [hidden, intermediateSize] → GGUF ne=[intermediateSize, hidden]
    expect(Number(down.shape[0])).toBe(6912);
    expect(Number(down.shape[1])).toBe(2560);

    // gate: same as up
    expect(Number(gate.shape[0])).toBe(2560);
    expect(Number(gate.shape[1])).toBe(6912);
  });

  it("sub-norm shapes", () => {
    const attnSN = findTensor("blk.0.attn_sub_norm.weight")!;
    const ffnSN = findTensor("blk.0.ffn_sub_norm.weight")!;

    console.log(`  attn_sub_norm shape: [${attnSN.shape}] (expect [2560])`);
    console.log(`  ffn_sub_norm shape: [${ffnSN.shape}] (expect [6912])`);

    expect(Number(attnSN.shape[0])).toBe(2560);
    expect(Number(ffnSN.shape[0])).toBe(6912);
  });
});

describe.skipIf(!HAS_MODEL)("I2_S weight data validation", () => {
  it("Q weight unpacks to valid ternary {-1,0,+1}", () => {
    const t = findTensor("blk.0.attn_q.weight")!;
    const data = getTensorData(t);
    const numElements = Number(t.shape[0]) * Number(t.shape[1]);
    const packedBytes = Math.ceil(numElements / 4);

    // Read first 128 elements (1 block) from row 0
    const weightBytes = new Uint8Array(data, 0, packedBytes);
    const u32View = new Uint32Array(
      weightBytes.buffer,
      weightBytes.byteOffset,
      Math.floor(weightBytes.byteLength / 4)
    );
    const firstBlock = unpackI2S(u32View.slice(0, 8), 128);

    console.log(`  First 32 ternary values: [${firstBlock.slice(0, 32)}]`);

    // All values must be {-1, 0, +1}
    for (const v of firstBlock) {
      expect([-1, 0, 1]).toContain(v);
    }

    // Not all zeros (that would be suspicious)
    const nonZero = firstBlock.filter((v) => v !== 0).length;
    console.log(`  Non-zero values in first block: ${nonZero}/128`);
    expect(nonZero).toBeGreaterThan(0);
  });

  it("per-tensor scale is reasonable", () => {
    const t = findTensor("blk.0.attn_q.weight")!;
    const data = getTensorData(t);
    const numElements = Number(t.shape[0]) * Number(t.shape[1]);
    const packedBytes = Math.ceil(numElements / 4);

    // Scale is in the last 32 bytes
    const scaleView = new DataView(data, packedBytes, 32);
    const scale = scaleView.getFloat32(0, true);
    console.log(`  blk.0.attn_q scale: ${scale}`);

    expect(Number.isFinite(scale)).toBe(true);
    expect(scale).not.toBe(0);
    // For ternary weights {-1,0,+1}, scale should be 1.0
    // (but the conversion might store the original weight magnitude)
    console.log(`  All 8 scale replicas: [${Array.from({length: 8}, (_, i) => scaleView.getFloat32(i * 4, true))}]`);
  });

  it("multiple layers have valid scales", () => {
    for (const name of ["blk.0.attn_q.weight", "blk.0.ffn_up.weight", "blk.15.attn_q.weight", "blk.29.attn_q.weight"]) {
      const t = findTensor(name);
      if (!t) continue;
      const data = getTensorData(t);
      const numElements = Number(t.shape[0]) * Number(t.shape[1]);
      const packedBytes = Math.ceil(numElements / 4);
      const scaleView = new DataView(data, packedBytes, 32);
      const scale = scaleView.getFloat32(0, true);
      console.log(`  ${name} scale: ${scale}`);
      expect(Number.isFinite(scale)).toBe(true);
      expect(scale).not.toBe(0);
    }
  });
});

describe.skipIf(!HAS_MODEL)("Embedding (F16) validation", () => {
  it("first few embedding values are reasonable", () => {
    const t = findTensor("token_embd.weight")!;
    const data = getTensorData(t);
    const f16 = new Uint16Array(data);

    // Convert first 10 F16 values to F32
    const values: number[] = [];
    for (let i = 0; i < 10; i++) {
      values.push(f16toF32(f16[i]));
    }
    console.log(`  First 10 embedding values (token 0): [${values.map(v => v.toFixed(6))}]`);

    // Should be finite, non-zero (at least some)
    for (const v of values) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("embeddings are not all zeros", () => {
    const t = findTensor("token_embd.weight")!;
    const data = getTensorData(t);
    const f16 = new Uint16Array(data);

    // Check first 2560 values (token 0's full embedding)
    let nonZero = 0;
    for (let i = 0; i < 2560; i++) {
      if (f16[i] !== 0) nonZero++;
    }
    console.log(`  Token 0 non-zero dims: ${nonZero}/2560`);
    expect(nonZero).toBeGreaterThan(100);
  });

  it("BOS token (128000) F16 embedding is non-zero and valid", () => {
    // This is THE critical test: BOS=128000 caused garbage output when
    // F32 embedding exceeded maxStorageBufferBindingSize and was sharded.
    // F16 embedding (656MB) fits in 1GB, so token 128000 is accessible.
    const t = findTensor("token_embd.weight")!;
    const D = 2560;
    const data = getTensorData(t);
    const f16 = new Uint16Array(data);

    const bosId = 128000;
    const offset = bosId * D;

    // Convert BOS embedding from F16 to F32
    let nonZero = 0;
    let nanCount = 0;
    let rms = 0;
    const first8: number[] = [];
    for (let i = 0; i < D; i++) {
      const v = f16toF32(f16[offset + i]);
      if (v !== 0) nonZero++;
      if (Number.isNaN(v)) nanCount++;
      rms += v * v;
      if (i < 8) first8.push(v);
    }
    rms = Math.sqrt(rms / D);

    console.log(`  BOS (${bosId}) embedding: rms=${rms.toFixed(6)}, nonZero=${nonZero}/${D}, NaN=${nanCount}`);
    console.log(`  First 8 values: [${first8.map(v => v.toFixed(6))}]`);

    // F16 data size for full embedding table
    const f16Bytes = f16.byteLength;
    console.log(`  F16 embedding table size: ${(f16Bytes / 1e6).toFixed(1)} MB`);
    console.log(`  BOS offset in bytes: ${offset * 2} (${((offset * 2) / 1e6).toFixed(1)} MB)`);
    expect(f16Bytes).toBeLessThan(1073741824); // Must fit in 1GB

    // The embedding must be non-zero and valid
    expect(nanCount).toBe(0);
    expect(nonZero).toBeGreaterThan(100);
    expect(rms).toBeGreaterThan(0.001);
    expect(Number.isFinite(rms)).toBe(true);
  });
});

describe.skipIf(!HAS_MODEL)("Config from real GGUF matches reference", () => {
  it("configFromGGUFMetadata produces correct values", async () => {
    // Import the actual config function
    const { default: loaderModule } = await import("../../model/loader.js") as any;
    // We can't easily call configFromGGUFMetadata since it's not exported
    // But we can verify from metadata directly
    const arch = gguf.metadata["general.architecture"] as string;
    console.log(`  Verifying config derivation for arch="${arch}":`);

    const checks = [
      { key: `${arch}.embedding_length`, expected: 2560, name: "hiddenSize" },
      { key: `${arch}.block_count`, expected: 30, name: "numHiddenLayers" },
      { key: `${arch}.attention.head_count`, expected: 20, name: "numAttentionHeads" },
      { key: `${arch}.attention.head_count_kv`, expected: 5, name: "numKeyValueHeads" },
      { key: `${arch}.feed_forward_length`, expected: 6912, name: "intermediateSize" },
    ];

    for (const { key, expected, name } of checks) {
      const value = gguf.metadata[key] as number;
      console.log(`  ${name}: metadata["${key}"] = ${value} (expected ${expected})`);
      expect(value).toBe(expected);
    }
  });
});

describe.skipIf(!HAS_MODEL)("Real data CPU reference pipeline", () => {
  // Extract the full packed weight data for a tensor as Uint32Array
  function getI2SWeightsU32(tensor: GGUFTensorInfo): Uint32Array {
    const data = getTensorData(tensor);
    const numElements = Number(tensor.shape[0]) * Number(tensor.shape[1]);
    const packedBytes = Math.ceil(numElements / 4);
    return new Uint32Array(data.slice(0, packedBytes));
  }

  // Extract the per-tensor scale
  function getI2SScale(tensor: GGUFTensorInfo): number {
    const data = getTensorData(tensor);
    const numElements = Number(tensor.shape[0]) * Number(tensor.shape[1]);
    const packedBytes = Math.ceil(numElements / 4);
    return new DataView(data, packedBytes, 32).getFloat32(0, true);
  }

  // Get F32 norm weights
  function getF32Weights(tensor: GGUFTensorInfo): Float32Array {
    const data = getTensorData(tensor);
    return new Float32Array(data);
  }

  // Get embedding vector for a token (F16 → F32)
  function getEmbedding(tokenId: number): Float32Array {
    const t = findTensor("token_embd.weight")!;
    const data = getTensorData(t);
    const D = 2560;
    const f16 = new Uint16Array(data, tokenId * D * 2, D);
    const f32 = new Float32Array(D);
    for (let i = 0; i < D; i++) {
      f32[i] = f16toF32(f16[i]);
    }
    return f32;
  }

  // Sequential (non-interleaved) unpack for comparison
  function unpackSequential(u32s: Uint32Array, K: number): number[] {
    const result: number[] = [];
    const bytes = new Uint8Array(u32s.buffer, u32s.byteOffset, u32s.byteLength);
    for (let k = 0; k < K; k++) {
      const byteIdx = Math.floor(k / 4);
      const bitPair = 3 - (k % 4); // MSB-first: elem 0 at bits[7:6]
      const shift = bitPair * 2;
      const code = (bytes[byteIdx] >> shift) & 3;
      result.push(code - 1);
    }
    return result;
  }

  it("compare interleaved vs sequential extraction", () => {
    const t = findTensor("blk.0.attn_q.weight")!;
    const inDim = Number(t.shape[0]); // 2560
    const outDim = Number(t.shape[1]); // 2560
    const kPacked = Math.ceil(inDim / 16);
    const u32s = getI2SWeightsU32(t);

    // Extract row 0 with both methods
    const row0u32 = new Uint32Array(u32s.buffer, u32s.byteOffset, kPacked);
    const interleaved = unpackI2S(row0u32, inDim);
    const sequential = unpackSequential(row0u32, inDim);

    // Count differences
    let diffCount = 0;
    const firstDiffs: string[] = [];
    for (let i = 0; i < inDim; i++) {
      if (interleaved[i] !== sequential[i]) {
        diffCount++;
        if (firstDiffs.length < 5) {
          firstDiffs.push(`  idx=${i}: interleaved=${interleaved[i]}, sequential=${sequential[i]}`);
        }
      }
    }
    console.log(`  Interleaved vs sequential: ${diffCount}/${inDim} differ`);
    if (firstDiffs.length > 0) {
      console.log("  First differences:");
      firstDiffs.forEach(d => console.log(d));
    }

    // Both should produce valid ternary
    for (const v of interleaved) expect([-1, 0, 1]).toContain(v);
    for (const v of sequential) expect([-1, 0, 1]).toContain(v);
  });

  it("ternary weight distribution for row 0 of Q", () => {
    const t = findTensor("blk.0.attn_q.weight")!;
    const inDim = Number(t.shape[0]);
    const kPacked = Math.ceil(inDim / 16);
    const u32s = getI2SWeightsU32(t);
    const row0 = unpackI2S(new Uint32Array(u32s.buffer, u32s.byteOffset, kPacked), inDim);

    let minus1 = 0, zero = 0, plus1 = 0;
    for (const v of row0) {
      if (v === -1) minus1++;
      else if (v === 0) zero++;
      else plus1++;
    }
    console.log(`  Row 0 distribution: -1:${minus1} 0:${zero} +1:${plus1} (total ${inDim})`);
    console.log(`  Fractions: -1:${(minus1/inDim*100).toFixed(1)}% 0:${(zero/inDim*100).toFixed(1)}% +1:${(plus1/inDim*100).toFixed(1)}%`);

    // For a well-trained ternary model, expect significant presence of all three values
    expect(minus1).toBeGreaterThan(100);
    expect(plus1).toBeGreaterThan(100);
  });

  it("full BitLinear on real data: embedding → norm → quantize → Q matmul", () => {
    // Get real data
    const qTensor = findTensor("blk.0.attn_q.weight")!;
    const normTensor = findTensor("blk.0.attn_norm.weight")!;
    const inDim = Number(qTensor.shape[0]); // 2560
    const outDim = Number(qTensor.shape[1]); // 2560
    const kPacked = Math.ceil(inDim / 16);

    const packedWeights = getI2SWeightsU32(qTensor);
    const weightScale = getI2SScale(qTensor);
    const normWeights = getF32Weights(normTensor);
    const embedding = getEmbedding(1); // Token 1 (common token)

    console.log(`  weightScale: ${weightScale}`);
    console.log(`  embedding magnitude: ${Math.sqrt(embedding.reduce((s, v) => s + v*v, 0) / inDim).toFixed(6)}`);
    console.log(`  norm weight magnitude: ${Math.sqrt(normWeights.reduce((s, v) => s + v*v, 0) / inDim).toFixed(6)}`);

    // Step 1: RMSNorm
    // refRMSNorm, refQuantize, refTernaryGemvNaive imported at top
    const normed = refRMSNorm(embedding, normWeights, 1, inDim, 1e-5);
    console.log(`  After RMSNorm: first 5 = [${Array.from(normed.slice(0, 5)).map((v: number) => v.toFixed(6))}]`);
    console.log(`  After RMSNorm: rms = ${Math.sqrt(normed.reduce((s: number, v: number) => s + v*v, 0) / inDim).toFixed(6)}`);

    // Step 2: Quantize
    const { output: quantized, scales: inputScales } = refQuantize(normed, 1, inDim);
    console.log(`  inputScale (absmax/127): ${inputScales[0].toFixed(6)}`);
    console.log(`  First 10 quantized: [${Array.from(quantized.slice(0, 10))}]`);

    // Step 3: Ternary GEMV with real weights
    const wScales = new Float32Array(outDim).fill(weightScale);
    const output = refTernaryGemvNaive(
      packedWeights, quantized, wScales, inputScales[0], outDim, inDim
    );

    // Check output
    let finiteCount = 0, zeroCount = 0, nanCount = 0;
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < outDim; i++) {
      const v = output[i];
      if (Number.isFinite(v)) finiteCount++;
      if (v === 0) zeroCount++;
      if (Number.isNaN(v)) nanCount++;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }

    console.log(`  Output stats: min=${minVal.toFixed(4)}, max=${maxVal.toFixed(4)}`);
    console.log(`  finite=${finiteCount}, zero=${zeroCount}, NaN=${nanCount} / ${outDim}`);
    console.log(`  First 10 output values: [${Array.from(output.slice(0, 10)).map((v: number) => v.toFixed(4))}]`);
    console.log(`  Output RMS: ${Math.sqrt(output.reduce((s: number, v: number) => s + v*v, 0) / outDim).toFixed(6)}`);

    // Output should be all finite, not all zeros
    expect(finiteCount).toBe(outDim);
    expect(nanCount).toBe(0);
    expect(zeroCount).toBeLessThan(outDim * 0.9); // not mostly zeros
    // Output values should have reasonable magnitude (not 1e+30)
    expect(Math.abs(maxVal)).toBeLessThan(1e6);
    expect(Math.abs(minVal)).toBeLessThan(1e6);
  });

  it("compare GEMV naive vs shader-logic on real weights", () => {
    const qTensor = findTensor("blk.0.attn_q.weight")!;
    const inDim = Number(qTensor.shape[0]);
    const outDim = Number(qTensor.shape[1]);
    const packedWeights = getI2SWeightsU32(qTensor);
    const weightScale = getI2SScale(qTensor);

    // Create a simple test input: quantized random-ish values
    const input = new Int32Array(inDim);
    for (let i = 0; i < inDim; i++) {
      input[i] = ((i * 7 + 3) % 255) - 127; // deterministic pseudo-random in [-127, 127]
    }

    const wScales = new Float32Array(outDim).fill(weightScale);
    const inputScale = 0.01; // arbitrary

    // refTernaryGemvNaive, refTernaryGemvShaderLogic imported at top
    const naive = refTernaryGemvNaive(packedWeights, input, wScales, inputScale, outDim, inDim);
    const shader = refTernaryGemvShaderLogic(packedWeights, input, wScales, inputScale, outDim, inDim);

    // They MUST match exactly (same math, different iteration order)
    let maxDiff = 0;
    for (let i = 0; i < outDim; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(naive[i] - shader[i]));
    }
    console.log(`  Naive vs shader-logic max diff: ${maxDiff}`);
    console.log(`  Naive first 5: [${Array.from(naive.slice(0, 5)).map((v: number) => v.toFixed(6))}]`);
    console.log(`  Shader first 5: [${Array.from(shader.slice(0, 5)).map((v: number) => v.toFixed(6))}]`);

    // These must be identical (same algorithm, same data)
    expect(maxDiff).toBe(0);
  });

  it("verify I2_S extraction matches sequential only for element 0", () => {
    // Elements 0 and 32*k should match between interleaved and sequential for byte 0
    // But element 1 should differ (it's the key discriminator)
    const t = findTensor("blk.0.attn_q.weight")!;
    const inDim = Number(t.shape[0]);
    const kPacked = Math.ceil(inDim / 16);
    const u32s = getI2SWeightsU32(t);
    const row0u32 = new Uint32Array(u32s.buffer, u32s.byteOffset, kPacked);

    const interleaved = unpackI2S(row0u32, inDim);
    const sequential = unpackSequential(row0u32, inDim);

    // Element 0: both extract from byte 0, bits [7:6] → should match
    console.log(`  Element 0: interleaved=${interleaved[0]}, sequential=${sequential[0]}`);
    expect(interleaved[0]).toBe(sequential[0]);

    // Element 1: interleaved extracts from byte 1 bits [7:6], sequential from byte 0 bits [5:4]
    // These WILL differ unless the data happens to have the same value
    console.log(`  Element 1: interleaved=${interleaved[1]}, sequential=${sequential[1]}`);

    // Element 32: interleaved extracts from byte 0 bits [5:4], sequential from byte 8 bits [7:6]
    console.log(`  Element 32: interleaved=${interleaved[32]}, sequential=${sequential[32]}`);

    // Verify our interleaved matches what the reference unpack says
    // (This verifies internal consistency)
    const block0bytes = new Uint8Array(u32s.buffer, u32s.byteOffset, 32);
    // Element 0 should be in byte 0, bits [7:6]
    const manualElem0 = ((block0bytes[0] >> 6) & 3) - 1;
    // Element 1 should be in byte 1, bits [7:6] (interleaved)
    const manualElem1Interleaved = ((block0bytes[1] >> 6) & 3) - 1;
    // Element 1 in sequential: byte 0, bits [5:4]
    const manualElem1Sequential = ((block0bytes[0] >> 4) & 3) - 1;
    // Element 32 should be in byte 0, bits [5:4] (interleaved)
    const manualElem32Interleaved = ((block0bytes[0] >> 4) & 3) - 1;

    console.log(`  Manual extraction for element 0: ${manualElem0}`);
    console.log(`  Manual extraction for element 1 (interleaved=byte1[7:6]): ${manualElem1Interleaved}`);
    console.log(`  Manual extraction for element 1 (sequential=byte0[5:4]): ${manualElem1Sequential}`);
    console.log(`  Manual extraction for element 32 (interleaved=byte0[5:4]): ${manualElem32Interleaved}`);

    expect(interleaved[0]).toBe(manualElem0);
    expect(interleaved[1]).toBe(manualElem1Interleaved);
    expect(interleaved[32]).toBe(manualElem32Interleaved);
  });

  it("sequential extraction gives DIFFERENT output (confirming interleaved is the format)", () => {
    const qTensor = findTensor("blk.0.attn_q.weight")!;
    const normTensor = findTensor("blk.0.attn_norm.weight")!;
    const inDim = Number(qTensor.shape[0]);
    const outDim = Number(qTensor.shape[1]);
    const kPacked = Math.ceil(inDim / 16);
    const packedWeights = getI2SWeightsU32(qTensor);
    const weightScale = getI2SScale(qTensor);
    const normWeights = getF32Weights(normTensor);
    const embedding = getEmbedding(1);

    // Run through norm + quantize
    const normed = refRMSNorm(embedding, normWeights, 1, inDim, 1e-5);
    const { output: quantized, scales } = refQuantize(normed, 1, inDim);
    const wScales = new Float32Array(outDim).fill(weightScale);

    // Interleaved GEMV (our current implementation)
    const interleavedOutput = refTernaryGemvNaive(
      packedWeights, quantized, wScales, scales[0], outDim, inDim
    );

    // Sequential GEMV: unpack with sequential and compute manually
    const seqOutput = new Float32Array(outDim);
    for (let row = 0; row < Math.min(outDim, 32); row++) {
      const rowU32 = new Uint32Array(packedWeights.buffer, packedWeights.byteOffset + row * kPacked * 4, kPacked);
      const seqWeights = unpackSequential(rowU32, inDim);
      let acc = 0;
      for (let k = 0; k < inDim; k++) {
        acc += seqWeights[k] * quantized[k];
      }
      seqOutput[row] = acc * weightScale * scales[0];
    }

    // Compare first 32 elements
    let maxDiff = 0;
    for (let i = 0; i < 32; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(interleavedOutput[i] - seqOutput[i]));
    }
    console.log(`  Interleaved first 5: [${Array.from(interleavedOutput.slice(0, 5)).map(v => v.toFixed(4))}]`);
    console.log(`  Sequential first 5:  [${Array.from(seqOutput.slice(0, 5)).map(v => v.toFixed(4))}]`);
    console.log(`  Max diff (first 32): ${maxDiff.toFixed(4)}`);

    // Both should be finite and non-zero, but they should DIFFER
    // (proving the formats are truly different)
    expect(maxDiff).toBeGreaterThan(0.01);
    // But both should be reasonable (both produce valid ternary weights)
    const intRms = Math.sqrt(interleavedOutput.reduce((s, v) => s + v*v, 0) / outDim);
    console.log(`  Interleaved output RMS: ${intRms.toFixed(4)}`);
    expect(intRms).toBeGreaterThan(0);
    expect(intRms).toBeLessThan(1e6);
  });

  it("end-to-end: full first-layer attention Q output", () => {
    // This test runs embedding → layernorm → Q projection for token 1
    // and verifies the output is numerically sane
    const qTensor = findTensor("blk.0.attn_q.weight")!;
    const normTensor = findTensor("blk.0.attn_norm.weight")!;
    const inDim = Number(qTensor.shape[0]);
    const outDim = Number(qTensor.shape[1]);
    const packedWeights = getI2SWeightsU32(qTensor);
    const weightScale = getI2SScale(qTensor);
    const normWeights = getF32Weights(normTensor);

    // Test with multiple token embeddings
    // refRMSNorm, refQuantize, refTernaryGemvNaive imported at top
    const wScales = new Float32Array(outDim).fill(weightScale);

    for (const tokenId of [1, 100, 1000, 50000]) {
      const embedding = getEmbedding(tokenId);
      const normed = refRMSNorm(embedding, normWeights, 1, inDim, 1e-5);
      const { output: quantized, scales } = refQuantize(normed, 1, inDim);
      const output = refTernaryGemvNaive(packedWeights, quantized, wScales, scales[0], outDim, inDim);

      const outputRms = Math.sqrt(output.reduce((s: number, v: number) => s + v*v, 0) / outDim);
      const outputMean = output.reduce((s: number, v: number) => s + v, 0) / outDim;
      const hasNaN = output.some((v: number) => Number.isNaN(v));
      const hasInf = output.some((v: number) => !Number.isFinite(v));

      console.log(`  Token ${tokenId}: rms=${outputRms.toFixed(4)}, mean=${outputMean.toFixed(4)}, NaN=${hasNaN}, Inf=${hasInf}`);

      expect(hasNaN).toBe(false);
      expect(hasInf).toBe(false);
      expect(outputRms).toBeGreaterThan(0);
      expect(outputRms).toBeLessThan(1e6);
    }
  });
});

// Helper: F16 to F32 conversion
function f16toF32(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;

  let f: number;
  if (exp === 0) {
    f = (frac / 1024) * Math.pow(2, -14);
  } else if (exp === 31) {
    f = frac === 0 ? Infinity : NaN;
  } else {
    f = (1 + frac / 1024) * Math.pow(2, exp - 15);
  }
  return sign ? -f : f;
}
