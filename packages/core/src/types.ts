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
  adapter: GPUAdapter;
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
}

export interface LoadProgress {
  phase: "download" | "parse" | "upload";
  loaded: number;
  total: number;
  /** 0.0 – 1.0 */
  fraction: number;
}

// ─── Generation ───

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repeatPenalty?: number;
  repeatLastN?: number;
  onToken?: (token: string) => void;
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

export interface GGUFMetadata {
  [key: string]: string | number | boolean | bigint | GGUFMetadata[];
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

export interface WorkerRequest {
  id: number;
  type: "load" | "generate" | "dispose";
  payload: unknown;
}

export interface WorkerResponse {
  id: number;
  type: "progress" | "token" | "done" | "error";
  payload: unknown;
}
