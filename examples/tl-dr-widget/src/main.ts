import { BitNet } from "0xbitnet";
import type { LoadProgress } from "0xbitnet";

// DOM
const setupSection = document.getElementById("setup-section")!;
const mainSection = document.getElementById("main-section")!;
const modelUrlInput = document.getElementById("model-url") as HTMLInputElement;
const loadBtn = document.getElementById("load-btn") as HTMLButtonElement;
const statusDiv = document.getElementById("status")!;
const inputText = document.getElementById("input-text") as HTMLTextAreaElement;
const maxTokensSelect = document.getElementById("max-tokens") as HTMLSelectElement;
const summarizeBtn = document.getElementById("summarize-btn") as HTMLButtonElement;
const summaryOutput = document.getElementById("summary-output")!;
const statsDiv = document.getElementById("stats")!;

let bitnet: BitNet | null = null;

// ─── Load Model ───

loadBtn.addEventListener("click", async () => {
  const url = modelUrlInput.value.trim();
  if (!url) return;

  loadBtn.disabled = true;
  statusDiv.innerHTML = `
    <span>Loading model...</span>
    <div class="progress-bar">
      <div class="progress-fill" id="progress-fill"></div>
    </div>
  `;

  try {
    bitnet = await BitNet.load(url, {
      onProgress: (progress: LoadProgress) => {
        const pct = (progress.fraction * 100).toFixed(1);
        const fill = document.getElementById("progress-fill");
        if (fill) fill.style.width = `${pct}%`;
        const span = statusDiv.querySelector("span");
        if (span) span.textContent = `${progress.phase}: ${pct}%`;
      },
    });

    setupSection.style.display = "none";
    mainSection.style.display = "flex";
    inputText.focus();
  } catch (err) {
    statusDiv.innerHTML = `<span style="color: #ef4444;">Error: ${(err as Error).message}</span>`;
    loadBtn.disabled = false;
  }
});

// ─── Summarize ───

summarizeBtn.addEventListener("click", summarize);

async function summarize(): Promise<void> {
  if (!bitnet) return;

  const text = inputText.value.trim();
  if (!text) return;

  summarizeBtn.disabled = true;
  summaryOutput.textContent = "";
  summaryOutput.classList.remove("empty");
  statsDiv.textContent = "";

  const maxTokens = parseInt(maxTokensSelect.value, 10);
  const prompt = `Summarize the following text concisely:\n\n${text}\n\nTL;DR:`;

  const startTime = performance.now();
  let tokenCount = 0;

  try {
    for await (const token of bitnet.generate(prompt, {
      maxTokens,
      temperature: 0.3,
      topK: 20,
    })) {
      summaryOutput.textContent += token;
      tokenCount++;
    }

    const elapsed = (performance.now() - startTime) / 1000;
    const tokPerSec = tokenCount / elapsed;
    statsDiv.textContent = `${tokenCount} tokens in ${elapsed.toFixed(1)}s (${tokPerSec.toFixed(1)} tok/s)`;
  } catch (err) {
    summaryOutput.textContent += `\n[Error: ${(err as Error).message}]`;
  }

  summarizeBtn.disabled = false;
}

// ─── WebGPU Check ───

if (!navigator.gpu) {
  loadBtn.disabled = true;
  statusDiv.innerHTML = `<span style="color: #ef4444;">WebGPU is not supported. Please use Chrome 113+ or Edge 113+.</span>`;
}
