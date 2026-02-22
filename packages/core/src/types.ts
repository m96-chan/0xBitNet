// ─── Model Configuration ───

export interface ModelConfig {
  modelType: "bitnet";
  vocabSize: number;
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  maxPositionEmbeddings: number;
  rmsNormEps: number;
  ropeTheta: number;
  tieWordEmbeddings: boolean;
  activation: "relu2" | "silu" | "swiglu";
  /** True when lm_head.weight is F16 (not I2_S ternary) */
  lmHeadF16?: boolean;
}

// ─── Weight Format ───

export type WeightFormat = "gguf" | "safetensors";

export interface TensorInfo {
  name: string;
  dtype: TensorDType;
  shape: number[];
  offset: number;
  size: number;
}

export type TensorDType =
  | "f32"
  | "f16"
  | "i8"
  | "i2" // ternary packed
  | "u8"
  | "i32";

// ─── GPU Types ───

export interface GPUContext {
  device: GPUDevice;
  adapter: GPUAdapter | null;
  limits: GPUSupportedLimits;
}

export interface ComputeKernel {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
}

export interface BufferEntry {
  buffer: GPUBuffer;
  size: number;
  inUse: boolean;
}

// ─── Loading ───

export interface LoadOptions {
  device?: GPUDevice;
  format?: WeightFormat;
  onProgress?: (progress: LoadProgress) => void;
  /** Abort signal to cancel the load operation. */
  signal?: AbortSignal;
}

export interface LoadProgress {
  phase: "download" | "parse" | "upload";
  loaded: number;
  total: number;
  /** 0.0 – 1.0 */
  fraction: number;
}

// ─── Chat ───

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

// ─── Generation ───

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  repeatPenalty?: number;
  repeatLastN?: number;
  onToken?: (token: string) => void;
  /** Abort signal to cancel generation early. */
  signal?: AbortSignal;
}

// ─── Diagnostics ───

/** Diagnostic result for a single pipeline stage. */
export interface DiagnosticResult {
  name: string;
  length: number;
  min: number;
  max: number;
  mean: number;
  rms: number;
  nanCount: number;
  infCount: number;
  zeroCount: number;
  first8: number[];
}

// ─── KV Cache ───

export interface KVCache {
  key: GPUBuffer;
  value: GPUBuffer;
  seqLen: number;
  maxSeqLen: number;
}

// ─── GGUF ───

export type GGUFMetadataValueType =
  | "uint8"
  | "int8"
  | "uint16"
  | "int16"
  | "uint32"
  | "int32"
  | "float32"
  | "uint64"
  | "int64"
  | "float64"
  | "bool"
  | "string"
  | "array";

export type GGUFMetadataValue =
  | string
  | number
  | boolean
  | bigint
  | GGUFMetadataValue[];

export interface GGUFMetadata {
  [key: string]: GGUFMetadataValue;
}

export interface GGUFTensorInfo {
  name: string;
  nDimensions: number;
  shape: bigint[];
  type: number;
  offset: bigint;
}

export interface GGUFHeader {
  magic: number;
  version: number;
  tensorCount: bigint;
  metadataKVCount: bigint;
}

export interface GGUFFile {
  header: GGUFHeader;
  metadata: GGUFMetadata;
  tensors: GGUFTensorInfo[];
  tensorDataOffset: number;
}

// ─── Safetensors ───

export interface SafetensorsHeader {
  [tensorName: string]: {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
  };
}

// ─── Tokenizer ───

export interface TokenizerConfig {
  type: "tiktoken" | "sentencepiece" | "bpe";
  vocabSize: number;
  bosToken?: number;
  eosToken?: number;
  padToken?: number;
}

// ─── Worker Messages ───

export type WorkerRequest =
  | { id: number; type: "load"; payload: { source: string; options?: LoadOptions } }
  | { id: number; type: "generate"; payload: { prompt: string | ChatMessage[]; options?: GenerateOptions } }
  | { id: number; type: "dispose" };

export type WorkerResponse =
  | { id: number; type: "progress"; payload: LoadProgress }
  | { id: number; type: "token"; payload: string }
  | { id: number; type: "done" }
  | { id: number; type: "error"; payload: { message: string; name?: string } };
