// Runs the Hugging Face model in a background thread so the page stays responsive.
import {
  pipeline,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";

let generator = null;
let loadedModel = null;

async function load(modelId) {
  if (generator && loadedModel === modelId) return;
  const device = (await hasWebGPU()) ? "webgpu" : "wasm";
  self.postMessage({ type: "status", text: `Loading model (${device === "webgpu" ? "GPU" : "CPU"})…` });

  const files = {};
  generator = await pipeline("text-generation", modelId, {
    // q4f16 halves memory on GPU; plain q4 is the safe choice on CPU.
    dtype: device === "webgpu" ? "q4f16" : "q4",
    device,
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        files[p.file] = { loaded: p.loaded, total: p.total };
        const all = Object.values(files);
        const loaded = all.reduce((s, f) => s + f.loaded, 0);
        const total = all.reduce((s, f) => s + f.total, 0);
        self.postMessage({ type: "progress", loaded, total });
      }
    },
  });
  loadedModel = modelId;
  self.postMessage({ type: "ready", device });
}

async function hasWebGPU() {
  try {
    if (!navigator.gpu) return false;
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await load(data.model);
    } else if (data.type === "chat") {
      await load(data.model);
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => self.postMessage({ type: "token", text }),
      });
      await generator(data.messages, {
        max_new_tokens: data.maxTokens || 256,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
        streamer,
      });
      self.postMessage({ type: "done" });
    }
  } catch (err) {
    self.postMessage({ type: "error", text: String(err?.message || err) });
  }
};
