import { create, globals } from "webgpu";
import { createInterface } from "readline/promises";
import { BitNet } from "0xbitnet";
import type { LoadProgress, ChatMessage } from "0xbitnet";

// ─── WebGPU Setup ───

// Inject WebGPU globals (GPUBufferUsage, GPUMapMode, GPUTextureUsage, etc.)
Object.assign(globalThis, globals);

const DEFAULT_URL =
  "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf";

// ─── CLI Argument Parsing ───

function parseArgs(): {
  url: string;
  maxTokens: number;
  temperature: number;
} {
  const args = process.argv.slice(2);
  let url = DEFAULT_URL;
  let maxTokens = 512;
  let temperature = 0.7;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--url":
        url = args[++i];
        break;
      case "--max-tokens":
        maxTokens = parseInt(args[++i], 10);
        break;
      case "--temperature":
        temperature = parseFloat(args[++i]);
        break;
      case "--help":
      case "-h":
        console.log(`0xBitNet Node.js CLI

Usage: npx tsx src/main.ts [options]

Options:
  --url <url>           Model GGUF URL (default: BitNet 2B-4T)
  --max-tokens <n>      Max tokens to generate (default: 512)
  --temperature <f>     Sampling temperature (default: 0.7)
  -h, --help            Show this help message`);
        process.exit(0);
    }
  }

  return { url, maxTokens, temperature };
}

// ─── Progress Display ───

function showProgress(progress: LoadProgress): void {
  const pct = (progress.fraction * 100).toFixed(1);
  const phase = progress.phase.padEnd(8);
  const bar = "█".repeat(Math.floor(progress.fraction * 30)).padEnd(30, "░");
  process.stderr.write(`\r  ${phase} [${bar}] ${pct}%`);
  if (progress.fraction >= 1) {
    process.stderr.write("\n");
  }
}

// ─── Main ───

async function main(): Promise<void> {
  const { url, maxTokens, temperature } = parseArgs();

  console.log("0xBitNet — Node.js CLI");
  console.log("======================\n");

  // Create Dawn WebGPU instance
  const gpu = create([]);
  const adapter = await gpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    console.error("Error: Failed to get WebGPU adapter.");
    return process.exit(1);
  }

  // Request device with max limits
  const requiredLimits: Record<string, number> = {};
  requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
  requiredLimits.maxStorageBufferBindingSize =
    adapter.limits.maxStorageBufferBindingSize;
  requiredLimits.maxStorageBuffersPerShaderStage =
    adapter.limits.maxStorageBuffersPerShaderStage;
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

  const device = await adapter.requestDevice({ requiredLimits });

  device.lost.then((info) => {
    console.error(`\nWebGPU device lost: ${info.message} (reason: ${info.reason})`);
    process.exit(1);
  });

  console.log(`GPU: ${adapter.info?.device || "Unknown"}`);
  console.log(`Loading model from:\n  ${url}\n`);

  // Load model
  const bitnet = await BitNet.load(url, {
    device,
    onProgress: showProgress,
  });

  console.log("\nModel loaded! Type your message (Ctrl+C to exit).\n");

  // Interactive chat loop
  const rl = createInterface({
    input: process.stdin,
    output: process.stderr,
  });

  const history: ChatMessage[] = [
    { role: "system", content: "You are a helpful assistant." },
  ];

  while (true) {
    let line: string;
    try {
      line = await rl.question("You: ");
    } catch {
      // EOF or Ctrl+C
      break;
    }

    const text = line.trim();
    if (!text) continue;

    history.push({ role: "user", content: text });

    process.stdout.write("\nAssistant: ");

    let tokenCount = 0;
    const t0 = performance.now();

    for await (const token of bitnet.generate(history, {
      maxTokens,
      temperature,
      topK: 40,
      repeatPenalty: 1.1,
    })) {
      process.stdout.write(token);
      tokenCount++;
    }

    const elapsed = (performance.now() - t0) / 1000;
    const tokSec = tokenCount / elapsed;

    process.stdout.write(
      `\n\n  [${tokenCount} tokens in ${elapsed.toFixed(1)}s — ${tokSec.toFixed(1)} tok/s]\n\n`
    );

    // Keep only the last user+assistant turn to avoid context overflow
    // (BitNet 2B-4T has 4096 token context)
    if (history.length > 5) {
      history.splice(1, history.length - 3);
    }
  }

  console.log("\nBye!");
  bitnet.dispose();
  rl.close();
  process.exit(0);
}

main().catch((err) => {
  console.error("\nFatal error:", err);
  process.exit(1);
});
