import type { SafetensorsHeader, TensorInfo, TensorDType } from "../types.js";

const HEADER_SIZE_BYTES = 8;

/**
 * Parse the header of a Safetensors file.
 * The header is a JSON blob preceded by an 8-byte little-endian u64 length.
 */
export function parseSafetensorsHeader(
  buffer: ArrayBuffer
): { header: SafetensorsHeader; dataOffset: number } {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerBytes = new Uint8Array(
    buffer,
    HEADER_SIZE_BYTES,
    headerLen
  );
  const headerStr = new TextDecoder().decode(headerBytes);
  const header: SafetensorsHeader = JSON.parse(headerStr);

  // Remove __metadata__ key if present (not a tensor)
  delete (header as Record<string, unknown>)["__metadata__"];

  const dataOffset = HEADER_SIZE_BYTES + headerLen;
  return { header, dataOffset };
}

/** Map safetensors dtype string to our TensorDType */
function mapDtype(dtype: string): TensorDType {
  switch (dtype) {
    case "F32":
      return "f32";
    case "F16":
      return "f16";
    case "I8":
      return "i8";
    case "I32":
      return "i32";
    case "U8":
      return "u8";
    default:
      throw new Error(`Unsupported safetensors dtype: ${dtype}`);
  }
}

/** Extract tensor info list from parsed header */
export function getTensorInfos(
  header: SafetensorsHeader,
  dataOffset: number
): TensorInfo[] {
  const infos: TensorInfo[] = [];
  for (const [name, meta] of Object.entries(header)) {
    const [start, end] = meta.data_offsets;
    infos.push({
      name,
      dtype: mapDtype(meta.dtype),
      shape: meta.shape,
      offset: dataOffset + start,
      size: end - start,
    });
  }
  return infos;
}
