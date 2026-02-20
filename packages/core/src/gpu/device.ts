import type { GPUContext } from "../types.js";

export class GPUDeviceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GPUDeviceError";
  }
}

/**
 * Initialize WebGPU adapter and device.
 * Requests maximum buffer sizes and compute invocations for large model support.
 *
 * @param existingDevice - Optional existing GPU device to reuse.
 * @returns GPU context with device, adapter, and limits.
 * @throws {GPUDeviceError} If WebGPU is not available or adapter request fails.
 *
 * @example
 * ```ts
 * const gpu = await initGPU();
 * console.log("Max buffer size:", gpu.limits.maxBufferSize);
 * ```
 */
export async function initGPU(existingDevice?: GPUDevice): Promise<GPUContext> {
  if (existingDevice) {
    return {
      device: existingDevice,
      adapter: null,
      limits: existingDevice.limits,
    };
  }

  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new GPUDeviceError(
      "WebGPU is not supported in this environment. " +
        "Please use a browser with WebGPU support (Chrome 113+, Edge 113+, Firefox Nightly)."
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    throw new GPUDeviceError(
      "Failed to obtain WebGPU adapter. Check that your GPU drivers are up to date."
    );
  }

  const requiredLimits: Record<string, number> = {};

  // Request max buffer sizes for large weight tensors
  requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
  requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;

  // Request max buffer count for bind groups
  const maxBuffersPerGroup =
    adapter.limits.maxStorageBuffersPerShaderStage;
  requiredLimits.maxStorageBuffersPerShaderStage = maxBuffersPerGroup;

  // Request max compute workgroup sizes
  requiredLimits.maxComputeWorkgroupSizeX =
    adapter.limits.maxComputeWorkgroupSizeX;
  requiredLimits.maxComputeWorkgroupSizeY =
    adapter.limits.maxComputeWorkgroupSizeY;
  requiredLimits.maxComputeWorkgroupSizeZ =
    adapter.limits.maxComputeWorkgroupSizeZ;
  requiredLimits.maxComputeInvocationsPerWorkgroup =
    adapter.limits.maxComputeInvocationsPerWorkgroup;
  requiredLimits.maxComputeWorkgroupStorageSize =
    adapter.limits.maxComputeWorkgroupStorageSize;

  const device = await adapter.requestDevice({
    requiredLimits,
  });

  device.lost.then((info) => {
    console.error(`WebGPU device lost: ${info.message} (reason: ${info.reason})`);
  });

  return { device, adapter, limits: device.limits };
}
