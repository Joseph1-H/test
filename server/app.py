"""JJ AI server: serves an open-source Hugging Face chat model over HTTP.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Configure with environment variables (see config below).
"""

import os
import threading

import torch
from fastapi import FastAPI, Header, HTTPException
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


def generate_reply(messages: list[dict], max_new_tokens: int | None, temperature: float) -> str:
    """messages: user/assistant turns. Adds the system prompt unless one is already first."""
    turns = [m for m in messages if m["role"] in ("user", "assistant")][-20:]
    system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
    conversation = [{"role": "system", "content": system}] + turns

    inputs = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    max_new = min(max_new_tokens or MAX_NEW_TOKENS, 1024)

    with generate_lock, torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if API_KEY and req.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is empty")
    reply = generate_reply([m.model_dump() for m in req.messages], req.max_new_tokens, req.temperature)
    return ChatResponse(reply=reply, model=MODEL_ID)


# ---- OpenAI-compatible endpoint ---------------------------------------------------
# Same format as llama.cpp's llama-server (used on the Pi Zero), so the app works with both.
class OAIMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class OAIRequest(BaseModel):
    messages: list[OAIMessage]
    max_tokens: int | None = None
    temperature: float = 0.7


@app.post("/v1/chat/completions")
def chat_completions(req: OAIRequest, authorization: str | None = Header(default=None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not any(m.role == "user" for m in req.messages):
        raise HTTPException(status_code=400, detail="messages has no user message")
    reply = generate_reply([m.model_dump() for m in req.messages], req.max_tokens, req.temperature)
    return {
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
    }
