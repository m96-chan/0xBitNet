import { chromium, type Browser } from "playwright";
import { createServer, type Server } from "node:http";

/**
 * Runs a shader on a real GPU and reads its outputs back.
 *
 * The tests beside this file check TypeScript references. Those references and
 * the shaders are two transcriptions of the same intent, and nothing made them
 * agree — this is what does.
 *
 * Two things about the environment, both learned the hard way:
 *
 * - `navigator.gpu` is absent on `about:blank`. WebGPU requires a secure
 *   context, so the page is served from 127.0.0.1. With that, **headless works**
 *   — the usual "WebGPU is not available headless" is normally this.
 * - Playwright's bundled Chromium does not ship WebGPU. The system Chrome does.
 */

/** One entry of the shader's `@group(0)` layout, in binding order. */
export type Binding =
  | { kind: "storage"; data: Float32Array | Int32Array | Uint32Array }
  | { kind: "out"; type: "f32" | "i32" | "u32"; length: number }
  | { kind: "uniform"; data: ArrayBuffer };

export interface Runner {
  /** Dispatches `code` and returns one array per `out` binding, in order. */
  run(options: {
    code: string;
    entry?: string;
    bindings: Binding[];
    workgroups: [number] | [number, number] | [number, number, number];
  }): Promise<(Float32Array | Int32Array | Uint32Array)[]>;
  close(): Promise<void>;
}

const CHROME = process.env.CHROME_PATH ?? "/opt/google/chrome/chrome";

/** Resolves to null when there is no usable GPU, so suites can skip. */
export async function createRunner(): Promise<Runner | null> {
  let server: Server | undefined;
  let browser: Browser | undefined;
  try {
    server = createServer((_, res) => {
      res.setHeader("Content-Type", "text/html");
      res.end("<!doctype html><title>wgsl</title>");
    });
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", () => resolve()));
    const { port } = server.address() as { port: number };

    browser = await chromium.launch({
      headless: true,
      executablePath: CHROME,
      args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
    });
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}/`);

    // One device for the whole suite: an adapter per kernel is slow, and
    // enough outstanding requests start failing outright.
    const ready = await page.evaluate(async () => {
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) return false;
      (globalThis as unknown as { __device: GPUDevice }).__device = await adapter.requestDevice();
      return true;
    });
    if (!ready) throw new Error("no adapter");

    return {
      async run({ code, entry = "main", bindings, workgroups }) {
        // Typed arrays do not survive the page boundary; bytes do.
        const wire = bindings.map((b) =>
          b.kind === "out"
            ? { kind: "out" as const, type: b.type, length: b.length }
            : {
                kind: b.kind,
                bytes: Array.from(
                  new Uint8Array(
                    b.kind === "uniform" ? b.data : b.data.buffer.slice(b.data.byteOffset, b.data.byteOffset + b.data.byteLength),
                  ),
                ),
              },
        );

        const results = await page.evaluate(
          async ({ code, entry, wire, workgroups }) => {
            const device = (globalThis as unknown as { __device: GPUDevice }).__device;
            const U = GPUBufferUsage;
            const outs: { index: number; buffer: GPUBuffer; bytes: number }[] = [];

            const buffers = wire.map((b, index) => {
              if (b.kind === "out") {
                const bytes = b.length * 4;
                const buffer = device.createBuffer({ size: bytes, usage: U.STORAGE | U.COPY_SRC });
                outs.push({ index, buffer, bytes });
                return buffer;
              }
              const data = new Uint8Array(b.bytes!);
              const usage = b.kind === "uniform" ? U.UNIFORM | U.COPY_DST : U.STORAGE | U.COPY_DST;
              // Uniform buffers have a 16-byte minimum binding size.
              const size = Math.max(b.kind === "uniform" ? 16 : 4, data.byteLength);
              const buffer = device.createBuffer({ size, usage });
              device.queue.writeBuffer(buffer, 0, data);
              return buffer;
            });

            const pipeline = device.createComputePipeline({
              layout: "auto",
              compute: { module: device.createShaderModule({ code }), entryPoint: entry },
            });
            const bindGroup = device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
            });

            const enc = device.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(...(workgroups as [number, number?, number?]));
            pass.end();

            const reads = outs.map(({ buffer, bytes }) => {
              const read = device.createBuffer({ size: bytes, usage: U.COPY_DST | U.MAP_READ });
              enc.copyBufferToBuffer(buffer, 0, read, 0, bytes);
              return read;
            });
            device.queue.submit([enc.finish()]);

            const out: number[][] = [];
            for (const read of reads) {
              await read.mapAsync(GPUMapMode.READ);
              out.push(Array.from(new Uint8Array(read.getMappedRange())));
            }
            return out;
          },
          { code, entry, wire, workgroups },
        );

        const outSpecs = bindings.filter((b): b is Extract<Binding, { kind: "out" }> => b.kind === "out");
        return results.map((bytes, i) => {
          const buffer = new Uint8Array(bytes).buffer;
          const type = outSpecs[i]!.type;
          if (type === "i32") return new Int32Array(buffer);
          if (type === "u32") return new Uint32Array(buffer);
          return new Float32Array(buffer);
        });
      },
      async close() {
        await browser?.close();
        server?.close();
      },
    };
  } catch {
    await browser?.close();
    server?.close();
    return null;
  }
}

/** Packs a params struct of mixed u32 / i32 / f32 into a uniform buffer. */
export function params(fields: ["u32" | "i32" | "f32", number][]): ArrayBuffer {
  const buffer = new ArrayBuffer(Math.max(16, fields.length * 4));
  const view = new DataView(buffer);
  fields.forEach(([kind, value], i) => {
    if (kind === "f32") view.setFloat32(i * 4, value, true);
    else if (kind === "i32") view.setInt32(i * 4, value, true);
    else view.setUint32(i * 4, value, true);
  });
  return buffer;
}
