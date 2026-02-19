import { BitNet } from "0xbitnet";
import type { LoadProgress } from "0xbitnet";

// DOM elements
const loadSection = document.getElementById("load-section")!;
const modelUrlInput = document.getElementById("model-url") as HTMLInputElement;
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

// ─── Load Model ───

loadBtn.addEventListener("click", async () => {
  const url = modelUrlInput.value.trim();
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

    addMessage("assistant", "Model loaded! How can I help you?");
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
    for await (const token of bitnet.generate(text, {
      maxTokens: 512,
      temperature: 0.7,
      topK: 40,
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

// ─── WebGPU Check ───

if (!navigator.gpu) {
  loadBtn.disabled = true;
  const p = document.createElement("p");
  p.style.color = "#ef4444";
  p.textContent =
    "WebGPU is not supported in your browser. Please use Chrome 113+ or Edge 113+.";
  loadSection.appendChild(p);
}
