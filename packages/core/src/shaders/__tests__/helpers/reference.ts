/**
 * CPU reference implementations for every shader.
 * Each function replicates the exact math of the corresponding WGSL shader.
 */
import { unpackI2S } from "./i2s-pack.js";

// ─── Embedding ───

export function refEmbedding(
  tokenIds: Uint32Array,
  embedTable: Float32Array,
  V: number,
  D: number
): Float32Array {
  const N = tokenIds.length;
  const output = new Float32Array(N * D);
  for (let n = 0; n < N; n++) {
    const id = tokenIds[n];
    for (let d = 0; d < D; d++) {
      output[n * D + d] = id < V ? embedTable[id * D + d] : 0;
    }
  }
  return output;
}

// ─── RMSNorm ───

export function refRMSNorm(
  input: Float32Array,
  weight: Float32Array,
  N: number,
  D: number,
  eps: number
): Float32Array {
  const output = new Float32Array(N * D);
  for (let row = 0; row < N; row++) {
    let sumSq = 0;
    for (let d = 0; d < D; d++) {
      const val = input[row * D + d];
      sumSq += val * val;
    }
    const rms = 1 / Math.sqrt(sumSq / D + eps);
    for (let d = 0; d < D; d++) {
      output[row * D + d] = input[row * D + d] * rms * weight[d];
    }
  }
  return output;
}

// ─── Quantize (absmax int8) ───

export function refQuantize(
  input: Float32Array,
  N: number,
  D: number
): { output: Int32Array; scales: Float32Array } {
  const output = new Int32Array(N * D);
  const scales = new Float32Array(N);

  for (let row = 0; row < N; row++) {
    let absmax = 0;
    for (let d = 0; d < D; d++) {
      absmax = Math.max(absmax, Math.abs(input[row * D + d]));
    }
    const scale = absmax === 0 ? 1.0 : absmax / 127.0;
    const invScale = absmax === 0 ? 0.0 : 127.0 / absmax;
    scales[row] = scale;

    for (let d = 0; d < D; d++) {
      const quantized = Math.round(input[row * D + d] * invScale);
      output[row * D + d] = Math.max(-127, Math.min(127, quantized));
    }
  }

  return { output, scales };
}

// ─── Ternary GEMV (naive: unpack then matmul) ───

export function refTernaryGemvNaive(
  packedWeights: Uint32Array,
  input: Int32Array,
  weightScales: Float32Array,
  inputScale: number,
  M: number,
  K: number
): Float32Array {
  const kPacked = Math.ceil(K / 16);
  const output = new Float32Array(M);

  for (let row = 0; row < M; row++) {
    const rowU32 = packedWeights.slice(row * kPacked, (row + 1) * kPacked);
    const weights = unpackI2S(rowU32, K);

    let acc = 0;
    for (let k = 0; k < K; k++) {
      acc += weights[k] * input[k];
    }
    output[row] = acc * weightScales[row] * inputScale;
  }

  return output;
}

// ─── Ternary GEMV (shader-logic: extract via same block/group/gp formula) ───

export function refTernaryGemvShaderLogic(
  packedWeights: Uint32Array,
  input: Int32Array,
  weightScales: Float32Array,
  inputScale: number,
  M: number,
  K: number
): Float32Array {
  const kPacked = Math.ceil(K / 16);
  const output = new Float32Array(M);

  for (let row = 0; row < M; row++) {
    const rowOffset = row * kPacked;
    let acc = 0;

    for (let col = 0; col < kPacked; col++) {
      const packed = packedWeights[rowOffset + col];
      const block = Math.floor(col / 8);
      const baseGp = (col % 8) * 4;

      for (let i = 0; i < 16; i++) {
        const byteInU32 = Math.floor(i / 4);
        const group = i % 4;
        const gp = baseGp + byteInU32;
        const kIdx = block * 128 + group * 32 + gp;
        if (kIdx < K) {
          const shift = byteInU32 * 8 + (6 - 2 * group);
          const code = (packed >>> shift) & 3;
          const w = code - 1;
          acc += w * input[kIdx];
        }
      }
    }

    output[row] = acc * weightScales[row] * inputScale;
  }

  return output;
}

// ─── Ternary GEMM (batched, shader-logic) ───
// Output[N, M] = Input[N, K] × Weights[M, K]^T

export function refTernaryGemm(
  packedWeights: Uint32Array,
  input: Int32Array,
  weightScales: Float32Array,
  inputScales: Float32Array,
  M: number,
  N: number,
  K: number
): Float32Array {
  const kPacked = Math.ceil(K / 16);
  const output = new Float32Array(N * M);

  for (let n = 0; n < N; n++) {
    for (let m = 0; m < M; m++) {
      const rowU32 = packedWeights.slice(m * kPacked, (m + 1) * kPacked);
      const weights = unpackI2S(rowU32, K);

      let acc = 0;
      for (let k = 0; k < K; k++) {
        acc += weights[k] * input[n * K + k];
      }
      const scale = weightScales[m] * inputScales[n];
      output[n * M + m] = acc * scale;
    }
  }

  return output;
}

// ─── RoPE ───

export function refRoPE(
  input: Float32Array,
  N: number,
  numHeads: number,
  headDim: number,
  posOffset: number,
  thetaBase: number
): Float32Array {
  const output = new Float32Array(input.length);
  const halfDim = headDim / 2;

  for (let token = 0; token < N; token++) {
    for (let head = 0; head < numHeads; head++) {
      for (let dp = 0; dp < halfDim; dp++) {
        const pos = token + posOffset;
        const freqExp = (-2 * dp) / headDim;
        const theta = pos * Math.pow(thetaBase, freqExp);
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);

        const baseIdx = (token * numHeads + head) * headDim + dp * 2;
        const x0 = input[baseIdx];
        const x1 = input[baseIdx + 1];
        output[baseIdx] = x0 * cosT - x1 * sinT;
        output[baseIdx + 1] = x0 * sinT + x1 * cosT;
      }
    }
  }

  return output;
}

// ─── Attention Scores: Q @ K^T * scale with causal mask ───

export function refAttentionScores(
  Q: Float32Array,
  K: Float32Array,
  N: number,
  S: number,
  numHeads: number,
  numKVHeads: number,
  headDim: number,
  scale: number
): Float32Array {
  const scores = new Float32Array(numHeads * N * S);
  const headsPerKV = numHeads / numKVHeads;

  for (let head = 0; head < numHeads; head++) {
    const kvHead = Math.floor(head / headsPerKV);
    for (let qPos = 0; qPos < N; qPos++) {
      for (let kPos = 0; kPos < S; kPos++) {
        let dot = 0;
        const qOffset = (qPos * numHeads + head) * headDim;
        const kOffset = (kPos * numKVHeads + kvHead) * headDim;
        for (let d = 0; d < headDim; d++) {
          dot += Q[qOffset + d] * K[kOffset + d];
        }
        const isCausal = kPos > qPos + (S - N);
        const score = isCausal ? -3.402823e38 : dot * scale;
        scores[(head * N + qPos) * S + kPos] = score;
      }
    }
  }

  return scores;
}

// ─── Attention V: attn_weights @ V ───

export function refAttentionV(
  attn: Float32Array,
  V: Float32Array,
  N: number,
  S: number,
  numHeads: number,
  numKVHeads: number,
  headDim: number
): Float32Array {
  const output = new Float32Array(N * numHeads * headDim);
  const headsPerKV = numHeads / numKVHeads;

  for (let qPos = 0; qPos < N; qPos++) {
    for (let head = 0; head < numHeads; head++) {
      const kvHead = Math.floor(head / headsPerKV);
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let s = 0; s < S; s++) {
          const a = attn[(head * N + qPos) * S + s];
          const v = V[(s * numKVHeads + kvHead) * headDim + d];
          sum += a * v;
        }
        output[(qPos * numHeads + head) * headDim + d] = sum;
      }
    }
  }

  return output;
}

// ─── Softmax ───

export function refSoftmax(
  input: Float32Array,
  N: number,
  D: number
): Float32Array {
  const output = new Float32Array(N * D);

  for (let row = 0; row < N; row++) {
    let maxVal = -Infinity;
    for (let d = 0; d < D; d++) {
      maxVal = Math.max(maxVal, input[row * D + d]);
    }
    let sum = 0;
    for (let d = 0; d < D; d++) {
      sum += Math.exp(input[row * D + d] - maxVal);
    }
    for (let d = 0; d < D; d++) {
      output[row * D + d] = Math.exp(input[row * D + d] - maxVal) / sum;
    }
  }

  return output;
}

// ─── Activation (ReLU², SiLU) ───

export function refActivation(
  input: Float32Array,
  activationType: number
): Float32Array {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    if (activationType === 0) {
      const relu = Math.max(0, x);
      output[i] = relu * relu;
    } else {
      output[i] = x / (1 + Math.exp(-x));
    }
  }
  return output;
}

// ─── Elementwise (add, multiply) ───

export function refElementwise(
  a: Float32Array,
  b: Float32Array,
  op: number
): Float32Array {
  const output = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    output[i] = op === 0 ? a[i] + b[i] : a[i] * b[i];
  }
  return output;
}

// ─── F32 Matmul (tied-embedding LM head) ───
// logits[n, v] = sum_d( hidden[n, d] * embed[v, d] )

export function refF32Matmul(
  hidden: Float32Array,
  embed: Float32Array,
  N: number,
  V: number,
  D: number
): Float32Array {
  const output = new Float32Array(N * V);
  for (let n = 0; n < N; n++) {
    for (let v = 0; v < V; v++) {
      let sum = 0;
      for (let d = 0; d < D; d++) {
        sum += hidden[n * D + d] * embed[v * D + d];
      }
      output[n * V + v] = sum;
    }
  }
  return output;
}

// ─── Dequantize ───

export function refDequantize(
  input: Int32Array,
  weightScale: number,
  inputScale: number
): Float32Array {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    output[i] = input[i] * weightScale * inputScale;
  }
  return output;
}
