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
 */
export async function initGPU(existingDevice?: GPUDevice): Promise<GPUContext> {
  if (existingDevice) {
    const adapter = null as unknown as GPUAdapter;
    return {
      device: existingDevice,
      adapter,
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

  // Request max buffer size for large weight tensors
  const maxBufferSize = adapter.limits.maxStorageBufferBindingSize;
  requiredLimits.maxStorageBufferBindingSize = maxBufferSize;

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
