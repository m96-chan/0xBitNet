import { BitNet, listCachedModels, deleteCachedModel } from "0xbitnet";
import type { LoadProgress, ChatMessage } from "0xbitnet";

const PRESET_MODELS = [
  {
    name: "BitNet 2B-4T",
    url: "https://huggingface.co/m96-chan/bitnet-b1.58-2B-4T-gguf/resolve/main/bitnet-b1.58-2B-4T.gguf",
    size: "~700 MB",
  },
];

// DOM elements
const loadSection = document.getElementById("load-section")!;
const cachedModelsDiv = document.getElementById("cached-models")!;
const loadBtn = document.getElementById("load-btn") as HTMLButtonElement;
const statusBar = document.getElementById("status-bar")!;
const statusText = document.getElementById("status-text")!;
const progressFill = document.getElementById("progress-fill")!;
const messagesDiv = document.getElementById("messages")!;
const inputArea = document.getElementById("input-area")!;
const userInput = document.getElementById("user-input") as HTMLInputElement;
const sendBtn = document.getElementById("send-btn") as HTMLButtonElement;

let bitnet: BitNet | null = null;
let isGenerating = false;

// ─── Cached Model Picker ───

function getSelectedModelUrl(): string {
  const selected = document.querySelector<HTMLInputElement>(
    'input[name="model-source"]:checked'
  );
  if (!selected || selected.value === "__new__") {
    const newUrlInput = document.getElementById("cached-new")
      ?.closest(".cached-item")
      ?.querySelector<HTMLInputElement>('input[type="text"]');
    return newUrlInput?.value.trim() ?? "";
  }
  return selected.value;
}

async function renderCachedModels(): Promise<void> {
  const cachedUrls = await listCachedModels();
  const cachedSet = new Set(cachedUrls);
  const presetUrls = new Set(PRESET_MODELS.map((p) => p.url));

  const list = document.createElement("div");
  list.className = "cached-list";

  let radioIndex = 0;

  // Presets first
  for (const preset of PRESET_MODELS) {
    const isCached = cachedSet.has(preset.url);
    const row = document.createElement("div");
    row.className = "cached-item";

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "model-source";
    radio.value = preset.url;
    radio.id = `cached-${radioIndex}`;
    if (radioIndex === 0) radio.checked = true;

    const label = document.createElement("label");
    label.htmlFor = `cached-${radioIndex}`;
    label.innerHTML = `${preset.name} <span class="preset-badge">Recommended</span> <span class="model-size">${preset.size}</span>`;

    row.append(radio, label);

    if (isCached) {
      const delBtn = document.createElement("button");
      delBtn.className = "delete-btn";
      delBtn.textContent = "Delete";
      delBtn.addEventListener("click", async (e) => {
        e.preventDefault();
        await deleteCachedModel(preset.url);
        await renderCachedModels();
      });
      row.appendChild(delBtn);
    }

    list.appendChild(row);
    radioIndex++;
  }

  // Non-preset cached models
  for (const url of cachedUrls) {
    if (presetUrls.has(url)) continue;

    const fileName = url.split("/").pop() || url;
    const row = document.createElement("div");
    row.className = "cached-item";

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "model-source";
    radio.value = url;
    radio.id = `cached-${radioIndex}`;

    const label = document.createElement("label");
    label.htmlFor = `cached-${radioIndex}`;
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
    radioIndex++;
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
  loadSection.style.display = "none";
  statusBar.style.display = "block";

  try {
    bitnet = await BitNet.load(url, {
      onProgress: (progress: LoadProgress) => {
        const pct = (progress.fraction * 100).toFixed(1);
        statusText.textContent = `${progress.phase}: ${pct}%`;
        progressFill.style.width = `${pct}%`;
      },
    });

    statusBar.style.display = "none";
    messagesDiv.style.display = "flex";
    inputArea.style.display = "flex";
    sendBtn.disabled = false;
    userInput.focus();

    addMessage("assistant", "Model loaded! Type a message or click 'Diagnose' to run GPU diagnostics.");

    // Refresh cached list (new model now appears)
    await renderCachedModels();

    // Run GPU diagnostic automatically after load
    runDiagnose();
  } catch (err) {
    statusText.textContent = `Error: ${(err as Error).message}`;
    progressFill.style.width = "0%";
    loadBtn.disabled = false;
    loadSection.style.display = "flex";
  }
});

// ─── Chat ───

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage(): Promise<void> {
  if (!bitnet || isGenerating) return;

  const text = userInput.value.trim();
  if (!text) return;

  userInput.value = "";
  addMessage("user", text);

  isGenerating = true;
  sendBtn.disabled = true;

  const assistantEl = addMessage("assistant", "");

  try {
    const messages: ChatMessage[] = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: text },
    ];
    for await (const token of bitnet.generate(messages, {
      maxTokens: 512,
      temperature: 0.7,
      topK: 40,
      repeatPenalty: 1.1,
    })) {
      assistantEl.textContent += token;
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
  } catch (err) {
    assistantEl.textContent += `\n[Error: ${(err as Error).message}]`;
  }

  isGenerating = false;
  sendBtn.disabled = false;
  userInput.focus();
}

function addMessage(role: "user" | "assistant", text: string): HTMLElement {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  messagesDiv.appendChild(el);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return el;
}

// ─── GPU Diagnostic ───

async function runDiagnose(): Promise<void> {
  if (!bitnet) return;

  const diagEl = addMessage("assistant", "Running GPU diagnostic...\n");

  try {
    const results = await bitnet.diagnose("Hello");

    let report = "=== GPU Diagnostic Results ===\n\n";
    for (const r of results) {
      report += `[${r.name}] len=${r.length}\n`;
      report += `  min=${r.min.toFixed(4)} max=${r.max.toFixed(4)} mean=${r.mean.toFixed(4)} rms=${r.rms.toFixed(4)}\n`;
      report += `  NaN=${r.nanCount} Inf=${r.infCount} zero=${r.zeroCount}\n`;
      report += `  first8: [${r.first8.map(v => v.toFixed(4)).join(", ")}]\n\n`;
    }

    diagEl.textContent = report;
    console.log("GPU Diagnostic:", results);
  } catch (err) {
    diagEl.textContent = `Diagnostic error: ${(err as Error).message}\n${(err as Error).stack}`;
    console.error("Diagnostic error:", err);
  }
}

// ─── Init ───

renderCachedModels();

// ─── WebGPU Check ───

if (!navigator.gpu) {
  loadBtn.disabled = true;
  const p = document.createElement("p");
  p.style.color = "#ef4444";
  p.textContent =
    "WebGPU is not supported in your browser. Please use Chrome 113+ or Edge 113+.";
  loadSection.appendChild(p);
}
