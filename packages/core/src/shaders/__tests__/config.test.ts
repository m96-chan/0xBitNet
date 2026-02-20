/**
 * Config validation tests.
 *
 * Verifies that model configuration matches the official HuggingFace
 * config.json values. This catches head count mismatches (the most
 * critical config error — wrong head_dim breaks all attention).
 */
import { describe, it, expect } from "vitest";
import {
  BITNET_2B_4T_CONFIG,
  headDim,
  gqaGroupSize,
} from "../../model/config.js";

describe("BitNet 2B-4T config", () => {
  // Official values from:
  // https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/main/config.json
  const HF_CONFIG = {
    vocab_size: 128256,
    hidden_size: 2560,
    intermediate_size: 6912,
    num_hidden_layers: 30,
    num_attention_heads: 20,
    num_key_value_heads: 5,
    max_position_embeddings: 4096,
    rms_norm_eps: 1e-5,
    rope_theta: 500000.0,
    tie_word_embeddings: true,
    hidden_act: "relu2",
  };

  it("vocab size matches", () => {
    expect(BITNET_2B_4T_CONFIG.vocabSize).toBe(HF_CONFIG.vocab_size);
  });

  it("hidden size matches", () => {
    expect(BITNET_2B_4T_CONFIG.hiddenSize).toBe(HF_CONFIG.hidden_size);
  });

  it("intermediate size matches", () => {
    expect(BITNET_2B_4T_CONFIG.intermediateSize).toBe(HF_CONFIG.intermediate_size);
  });

  it("num hidden layers matches", () => {
    expect(BITNET_2B_4T_CONFIG.numHiddenLayers).toBe(HF_CONFIG.num_hidden_layers);
  });

  it("num attention heads matches (CRITICAL: 20, not 32!)", () => {
    expect(BITNET_2B_4T_CONFIG.numAttentionHeads).toBe(HF_CONFIG.num_attention_heads);
  });

  it("num key-value heads matches (CRITICAL: 5, not 8!)", () => {
    expect(BITNET_2B_4T_CONFIG.numKeyValueHeads).toBe(HF_CONFIG.num_key_value_heads);
  });

  it("head_dim = hidden_size / num_attention_heads = 128 (NOT 80!)", () => {
    expect(headDim(BITNET_2B_4T_CONFIG)).toBe(128);
    expect(headDim(BITNET_2B_4T_CONFIG)).toBe(
      HF_CONFIG.hidden_size / HF_CONFIG.num_attention_heads
    );
  });

  it("GQA group size = num_heads / num_kv_heads = 4", () => {
    expect(gqaGroupSize(BITNET_2B_4T_CONFIG)).toBe(4);
  });

  it("rope theta matches", () => {
    expect(BITNET_2B_4T_CONFIG.ropeTheta).toBe(HF_CONFIG.rope_theta);
  });

  it("RMS norm epsilon matches", () => {
    expect(BITNET_2B_4T_CONFIG.rmsNormEps).toBe(HF_CONFIG.rms_norm_eps);
  });

  it("tie word embeddings is true", () => {
    expect(BITNET_2B_4T_CONFIG.tieWordEmbeddings).toBe(HF_CONFIG.tie_word_embeddings);
  });

  it("activation is relu2", () => {
    expect(BITNET_2B_4T_CONFIG.activation).toBe("relu2");
  });

  it("Q projection dimension is consistent", () => {
    // Q: [hidden, numHeads * headDim] = [2560, 20*128] = [2560, 2560]
    const qOutDim = BITNET_2B_4T_CONFIG.numAttentionHeads * headDim(BITNET_2B_4T_CONFIG);
    expect(qOutDim).toBe(BITNET_2B_4T_CONFIG.hiddenSize);
  });

  it("K/V projection dimension is consistent", () => {
    // K/V: [hidden, numKVHeads * headDim] = [2560, 5*128] = [2560, 640]
    const kvOutDim = BITNET_2B_4T_CONFIG.numKeyValueHeads * headDim(BITNET_2B_4T_CONFIG);
    expect(kvOutDim).toBe(640);
  });

  it("max position embeddings matches", () => {
    expect(BITNET_2B_4T_CONFIG.maxPositionEmbeddings).toBe(HF_CONFIG.max_position_embeddings);
  });
});

describe("dimension consistency checks", () => {
  it("head_dim divides hidden_size evenly", () => {
    expect(BITNET_2B_4T_CONFIG.hiddenSize % BITNET_2B_4T_CONFIG.numAttentionHeads).toBe(0);
  });

  it("num_attention_heads divisible by num_kv_heads (GQA)", () => {
    expect(BITNET_2B_4T_CONFIG.numAttentionHeads % BITNET_2B_4T_CONFIG.numKeyValueHeads).toBe(0);
  });

  it("head_dim is even (required for RoPE pairs)", () => {
    expect(headDim(BITNET_2B_4T_CONFIG) % 2).toBe(0);
  });
});
