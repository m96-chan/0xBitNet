import type {
  GGUFFile,
  GGUFHeader,
  GGUFMetadata,
  GGUFTensorInfo,
} from "../types.js";

// GGUF magic: "GGUF" in little-endian
const GGUF_MAGIC = 0x46475547;

// GGUF metadata value type enum
const enum MetaType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

// GGML quantization types we care about
export const GGML_TYPE_F32 = 0;
export const GGML_TYPE_F16 = 1;
export const GGML_TYPE_I8 = 16;
export const GGML_TYPE_I16 = 17;
export const GGML_TYPE_I32 = 18;
export const GGML_TYPE_I2_S = 27; // 2-bit ternary (packed)

/** Bytes per element for GGML types (fractional for sub-byte) */
export function ggmlTypeSize(type: number): number {
  switch (type) {
    case GGML_TYPE_F32:
      return 4;
    case GGML_TYPE_F16:
      return 2;
    case GGML_TYPE_I8:
      return 1;
    case GGML_TYPE_I16:
      return 2;
    case GGML_TYPE_I32:
      return 4;
    case GGML_TYPE_I2_S:
      return 0.25; // 2 bits per value = 0.25 bytes
    default:
      throw new Error(`Unsupported GGML type: ${type}`);
  }
}

/** Block size for quantized types */
export function ggmlBlockSize(type: number): number {
  switch (type) {
    case GGML_TYPE_I2_S:
      return 32; // 32 values per block (8 bytes packed + scale)
    default:
      return 1;
  }
}

/**
 * Streaming GGUF parser.
 * Reads header, metadata, and tensor info from a GGUF file.
 */
export class GGUFParser {
  private view: DataView;
  private offset: number;
  private textDecoder = new TextDecoder("utf-8");

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
    this.offset = 0;
  }

  /** Parse the entire GGUF file structure */
  parse(): GGUFFile {
    const header = this.readHeader();
    const metadata = this.readMetadata(Number(header.metadataKVCount));
    const tensors = this.readTensorInfos(Number(header.tensorCount));

    // Tensor data starts after alignment padding
    const alignment = (metadata["general.alignment"] as number) || 32;
    const tensorDataOffset =
      Math.ceil(this.offset / alignment) * alignment;

    return { header, metadata, tensors, tensorDataOffset };
  }

  private readHeader(): GGUFHeader {
    const magic = this.readU32();
    if (magic !== GGUF_MAGIC) {
      throw new Error(
        `Invalid GGUF magic: 0x${magic.toString(16)} (expected 0x${GGUF_MAGIC.toString(16)})`
      );
    }
    const version = this.readU32();
    if (version < 2 || version > 3) {
      throw new Error(`Unsupported GGUF version: ${version}`);
    }
    const tensorCount = this.readU64();
    const metadataKVCount = this.readU64();
    return { magic, version, tensorCount, metadataKVCount };
  }

  private readMetadata(count: number): GGUFMetadata {
    const metadata: GGUFMetadata = {};
    for (let i = 0; i < count; i++) {
      const key = this.readString();
      const value = this.readMetadataValue();
      metadata[key] = value;
    }
    return metadata;
  }

  private readMetadataValue(): string | number | boolean | bigint | GGUFMetadata[] {
    const valueType = this.readU32();
    return this.readValueOfType(valueType);
  }

  private readValueOfType(
    type: number
  ): string | number | boolean | bigint | GGUFMetadata[] {
    switch (type) {
      case MetaType.UINT8:
        return this.readU8();
      case MetaType.INT8:
        return this.readI8();
      case MetaType.UINT16:
        return this.readU16();
      case MetaType.INT16:
        return this.readI16();
      case MetaType.UINT32:
        return this.readU32();
      case MetaType.INT32:
        return this.readI32();
      case MetaType.FLOAT32:
        return this.readF32();
      case MetaType.BOOL:
        return this.readU8() !== 0;
      case MetaType.STRING:
        return this.readString();
      case MetaType.UINT64:
        return this.readU64();
      case MetaType.INT64:
        return this.readI64();
      case MetaType.FLOAT64:
        return this.readF64();
      case MetaType.ARRAY: {
        const elemType = this.readU32();
        const len = Number(this.readU64());
        const arr: GGUFMetadata[] = [];
        for (let i = 0; i < len; i++) {
          arr.push({
            value: this.readValueOfType(elemType),
          } as unknown as GGUFMetadata);
        }
        return arr;
      }
      default:
        throw new Error(`Unknown GGUF metadata type: ${type}`);
    }
  }

  private readTensorInfos(count: number): GGUFTensorInfo[] {
    const tensors: GGUFTensorInfo[] = [];
    for (let i = 0; i < count; i++) {
      const name = this.readString();
      const nDimensions = this.readU32();
      const shape: bigint[] = [];
      for (let d = 0; d < nDimensions; d++) {
        shape.push(this.readU64());
      }
      const type = this.readU32();
      const offset = this.readU64();
      tensors.push({ name, nDimensions, shape, type, offset });
    }
    return tensors;
  }

  // ─── Primitive readers ───

  private readU8(): number {
    const v = this.view.getUint8(this.offset);
    this.offset += 1;
    return v;
  }

  private readI8(): number {
    const v = this.view.getInt8(this.offset);
    this.offset += 1;
    return v;
  }

  private readU16(): number {
    const v = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return v;
  }

  private readI16(): number {
    const v = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return v;
  }

  private readU32(): number {
    const v = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return v;
  }

  private readI32(): number {
    const v = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return v;
  }

  private readU64(): bigint {
    const v = this.view.getBigUint64(this.offset, true);
    this.offset += 8;
    return v;
  }

  private readI64(): bigint {
    const v = this.view.getBigInt64(this.offset, true);
    this.offset += 8;
    return v;
  }

  private readF32(): number {
    const v = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return v;
  }

  private readF64(): number {
    const v = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return v;
  }

  private readString(): string {
    const len = Number(this.readU64());
    const bytes = new Uint8Array(this.view.buffer, this.offset, len);
    this.offset += len;
    return this.textDecoder.decode(bytes);
  }
}
