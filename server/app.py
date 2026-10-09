"""JJ AI server: serves an open-source Hugging Face chat model over HTTP.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Configure with environment variables (see config below).
"""

import os
import threading

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Config -----------------------------------------------------------------
# A small, Apache-2.0 licensed instruct model that can run on a Raspberry Pi CPU.
MODEL_ID = os.getenv("JJ_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
SYSTEM_PROMPT = os.getenv(
    "JJ_SYSTEM_PROMPT",
    "You are JJ, a helpful, friendly AI assistant. Answer clearly and concisely.",
)
MAX_NEW_TOKENS = int(os.getenv("JJ_MAX_NEW_TOKENS", "256"))
API_KEY = os.getenv("JJ_API_KEY", "")  # optional: require this key from clients
THREADS = int(os.getenv("JJ_THREADS", str(os.cpu_count() or 4)))

torch.set_num_threads(THREADS)

# ---- Model ------------------------------------------------------------------
print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
generate_lock = threading.Lock()  # one generation at a time on small hardware
print("Model ready.")

# ---- API --------------------------------------------------------------------
app = FastAPI(title="JJ AI")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int | None = None
    temperature: float = 0.7
    api_key: str | None = None


class ChatResponse(BaseModel):
    reply: str
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if API_KEY and req.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is empty")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation += [m.model_dump() for m in req.messages[-20:]]  # keep context small

    inputs = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    max_new = min(req.max_new_tokens or MAX_NEW_TOKENS, 1024)

    with generate_lock, torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    reply = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
    return ChatResponse(reply=reply.strip(), model=MODEL_ID)
