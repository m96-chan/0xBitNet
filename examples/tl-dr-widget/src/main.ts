import { BitNet, listCachedModels, deleteCachedModel } from "0xbitnet";
import type { LoadProgress, ChatMessage } from "0xbitnet";

// DOM
const setupSection = document.getElementById("setup-section")!;
const mainSection = document.getElementById("main-section")!;
const cachedModelsDiv = document.getElementById("cached-models")!;
const modelUrlInput = document.getElementById("model-url") as HTMLInputElement;
const loadBtn = document.getElementById("load-btn") as HTMLButtonElement;
const statusDiv = document.getElementById("status")!;
const inputText = document.getElementById("input-text") as HTMLTextAreaElement;
const maxTokensSelect = document.getElementById("max-tokens") as HTMLSelectElement;
const summarizeBtn = document.getElementById("summarize-btn") as HTMLButtonElement;
const summaryOutput = document.getElementById("summary-output")!;
const statsDiv = document.getElementById("stats")!;

let bitnet: BitNet | null = null;

// ─── Cached Model Picker ───

function getSelectedModelUrl(): string {
  const selected = document.querySelector<HTMLInputElement>(
    'input[name="model-source"]:checked'
  );
  if (!selected || selected.value === "__new__") {
    return modelUrlInput.value.trim();
  }
  return selected.value;
}

async function renderCachedModels(): Promise<void> {
  const urls = await listCachedModels();

  if (urls.length === 0) {
    cachedModelsDiv.style.display = "none";
    modelUrlInput.style.display = "";
    return;
  }

  cachedModelsDiv.style.display = "";
  modelUrlInput.style.display = "none";

  const list = document.createElement("div");
  list.className = "cached-list";

  for (let i = 0; i < urls.length; i++) {
    const url = urls[i];
    const fileName = url.split("/").pop() || url;

    const row = document.createElement("div");
    row.className = "cached-item";

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "model-source";
    radio.value = url;
    radio.id = `cached-${i}`;
    if (i === 0) radio.checked = true;

    const label = document.createElement("label");
    label.htmlFor = `cached-${i}`;
    label.textContent = fileName;
    label.title = url;

    const delBtn = document.createElement("button");
    delBtn.className = "delete-btn";
    delBtn.textContent = "Delete";
    delBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      await deleteCachedModel(url);
      await renderCachedModels();
    });

    row.append(radio, label, delBtn);
    list.appendChild(row);
  }

  // "New URL" option
  const newRow = document.createElement("div");
  newRow.className = "cached-item";

  const newRadio = document.createElement("input");
  newRadio.type = "radio";
  newRadio.name = "model-source";
  newRadio.value = "__new__";
  newRadio.id = "cached-new";

  const newLabel = document.createElement("label");
  newLabel.htmlFor = "cached-new";
  newLabel.textContent = "New URL:";
  newLabel.style.flex = "none";

  const newInput = document.createElement("input");
  newInput.type = "text";
  newInput.placeholder = "https://huggingface.co/.../model.gguf";
  newInput.addEventListener("focus", () => {
    newRadio.checked = true;
  });
  newInput.addEventListener("input", () => {
    modelUrlInput.value = newInput.value;
  });

  newRow.append(newRadio, newLabel, newInput);
  list.appendChild(newRow);

  cachedModelsDiv.innerHTML = "";
  cachedModelsDiv.appendChild(list);
}

// ─── Load Model ───

loadBtn.addEventListener("click", async () => {
  const url = getSelectedModelUrl();
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

    await renderCachedModels();
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
  const messages: ChatMessage[] = [
    { role: "system", content: "You are a helpful assistant. Summarize the user's text concisely." },
    { role: "user", content: `Summarize the following text:\n\n${text}` },
  ];

  const startTime = performance.now();
  let tokenCount = 0;

  try {
    for await (const token of bitnet.generate(messages, {
      maxTokens,
      temperature: 0.3,
      topK: 20,
      repeatPenalty: 1.1,
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

// ─── Init ───

renderCachedModels();

// ─── WebGPU Check ───

if (!navigator.gpu) {
  loadBtn.disabled = true;
  statusDiv.innerHTML = `<span style="color: #ef4444;">WebGPU is not supported. Please use Chrome 113+ or Edge 113+.</span>`;
}
