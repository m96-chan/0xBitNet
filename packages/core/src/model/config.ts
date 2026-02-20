import type { ModelConfig } from "../types.js";

/** Default config for microsoft/bitnet-b1.58-2B-4T */
export const BITNET_2B_4T_CONFIG: ModelConfig = {
  modelType: "bitnet",
  vocabSize: 128256,
  hiddenSize: 2560,
  intermediateSize: 6912,
  numHiddenLayers: 30,
  numAttentionHeads: 20,
  numKeyValueHeads: 5,
  maxPositionEmbeddings: 4096,
  rmsNormEps: 1e-5,
  ropeTheta: 500000.0,
  tieWordEmbeddings: true,
  activation: "relu2",
};

/** Default config for 1bitLLM/bitnet_b1_58-large (0.7B) */
export const BITNET_0_7B_CONFIG: ModelConfig = {
  modelType: "bitnet",
  vocabSize: 32002,
  hiddenSize: 2048,
  intermediateSize: 5632,
  numHiddenLayers: 24,
  numAttentionHeads: 32,
  numKeyValueHeads: 32,
  maxPositionEmbeddings: 2048,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  tieWordEmbeddings: false,
  activation: "silu",
};

/**
 * Default config for HF1BitLLM/bitnet_b1_58-3B.
 * @deprecated The upstream HuggingFace model has been removed. This config is kept for backward compatibility.
 */
export const BITNET_3B_CONFIG: ModelConfig = {
  modelType: "bitnet",
  vocabSize: 32002,
  hiddenSize: 3200,
  intermediateSize: 8640,
  numHiddenLayers: 26,
  numAttentionHeads: 32,
  numKeyValueHeads: 32,
  maxPositionEmbeddings: 2048,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  tieWordEmbeddings: false,
  activation: "silu",
};

/** Derive head dimension from config */
export function headDim(config: ModelConfig): number {
  return config.hiddenSize / config.numAttentionHeads;
}

/** Derive GQA group size */
export function gqaGroupSize(config: ModelConfig): number {
  return config.numAttentionHeads / config.numKeyValueHeads;
}
