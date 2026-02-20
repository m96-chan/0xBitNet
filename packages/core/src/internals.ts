/**
 * 0xBitNet internal building blocks.
 *
 * These are low-level primitives for advanced users who need direct control
 * over the inference pipeline. Most users should use the high-level `BitNet`
 * class from the main `"0xbitnet"` entry point instead.
 *
 * @example
 * ```ts
 * import { BitLinear, Attention, PipelineManager } from "0xbitnet/internals";
 * ```
 *
 * @packageDocumentation
 */

export type {
  ComputeKernel,
  BufferEntry,
  GGUFMetadataValueType,
  GGUFMetadataValue,
  GGUFTensorInfo,
  GGUFHeader,
  GGUFFile,
  TensorInfo,
  TensorDType,
  SafetensorsHeader,
  KVCache,
} from "./types.js";

export { PipelineManager } from "./gpu/pipeline.js";
export { BufferPool } from "./gpu/buffer-pool.js";
export { GGUFParser } from "./model/gguf.js";
export { parseSafetensorsHeader, getTensorInfos } from "./model/safetensors.js";
export { BitLinear } from "./nn/bitlinear.js";
export { Attention, createKVCache } from "./nn/attention.js";
export { FFN } from "./nn/ffn.js";
export { TransformerBlock } from "./nn/transformer.js";
