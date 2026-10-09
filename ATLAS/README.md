# ATLAS

**Automated Tasks, Logic and Software**: a local-first AI coding assistant.

ATLAS runs an open-weight coding model on your own GPU through [Ollama](https://ollama.com). A cloud model (any OpenAI-compatible API) is optional, off by default, and always asks before anything leaves your machine.

> **Status: v0.1, Phase 1.** ATLAS can chat with local and cloud models. It cannot read or change files yet; that's Phase 2. The model is told this, so it won't pretend to have edited files.

## Install (Linux Mint / Ubuntu)

Requirements: Python 3.10+, an NVIDIA driver (Driver Manager), ~6 GB free disk for the model.

```bash
# 1. Get the code (look at what's in /data/ATLAS first: cp -n never overwrites existing files)
ls -la /data/ATLAS
git clone https://github.com/Joseph1-H/test.git /tmp/atlas-src
mkdir -p /data/ATLAS && cp -rn /tmp/atlas-src/ATLAS/. /data/ATLAS/

# 2. Set up (asks before installing Ollama or downloading the model)
cd /data/ATLAS
bash setup.sh
```

`setup.sh` creates `.venv`, installs ATLAS, creates `.env`, checks your GPU and Ollama, offers to download the model, runs the unit tests, runs `atlas doctor`, and runs a real inference test against the local model.

## Use

```bash
/data/ATLAS/.venv/bin/atlas --workspace /data/projects/example   # interactive
/data/ATLAS/.venv/bin/atlas ask "Write a Python function that reverses a list"
/data/ATLAS/.venv/bin/atlas doctor                                 # check GPU / Ollama / model
```
If you let `setup.sh` link it into `~/.local/bin`, plain `atlas` works from anywhere.

| Command | What it does |
|---|---|
| `/help` | List commands |
| `/model` | List models; `/model NAME` switches |
| `/local` | Use the local Ollama model |
| `/cloud` | Use the cloud model (asks permission first) |
| `/status` | Active model, cloud state, workspace |
| `/doctor` | Environment checks |
| `/clear` | Forget the conversation |
| `/history` | Show the conversation |
| `/exit` | Quit (or Ctrl+D). Ctrl+C stops a reply. |

## Choosing a model (RTX 4050, 6 GB VRAM)

| Model | Size | License | Notes |
|---|---|---|---|
| `qwen2.5-coder:7b` | 4.7 GB | Apache-2.0 | **Default.** Best fit for 6 GB. |
| `qwen2.5-coder:1.5b` | 1 GB | Apache-2.0 | Fast fallback; weaker. |
| `qwen2.5-coder:3b` | 1.9 GB | Qwen Research (non-commercial) | Avoid if you may sell ATLAS. |

Switch with `ATLAS_LOCAL_MODEL` in `.env` or `/model NAME`. Any Ollama model works, including Hugging Face GGUF models (`ollama pull hf.co/<user>/<repo>`). Hugging Face models served by vLLM, TGI or llama.cpp can be used through the OpenAI-compatible provider pointed at `http://localhost:...` (treated as local, no consent prompt).

## Cloud (optional)

In `.env`: set `ATLAS_CLOUD_BASE_URL`, `ATLAS_CLOUD_MODEL`, `ATLAS_CLOUD_API_KEY`, and `ATLAS_CLOUD_ALLOWED=true`. Then `/cloud` in a session shows exactly what will be sent and where, and asks. The key is only read from `.env`/environment, never printed or logged. `setup.sh` makes `.env` owner-readable only, and `atlas doctor` warns if it isn't.

## Architecture

```
src/atlas/
  cli.py              terminal UI, slash commands, `ask` / `doctor` subcommands
  session.py          conversation history (temporary), active model, cloud consent
  providers/
    base.py           ModelProvider interface + Message / ChatResult
    ollama.py         local inference via Ollama's HTTP API (streaming)
    openai_compat.py  any OpenAI-compatible endpoint (streaming SSE)
    http.py           stdlib HTTP helpers (no third-party HTTP client)
  config.py           settings from .env, validated
  doctor.py           environment checks
  logging_setup.py    JSON-lines logs in ~/.local/share/atlas/atlas.log
```

**Adding a provider:** subclass `ModelProvider` (implement `chat`, `list_models`, `health`) and add a factory in `providers/__init__.py`. Nothing else changes.

Design decisions:
- **Standard-library HTTP:** one less dependency, and streaming is simple line reading. The only runtime dependency is `python-dotenv`.
- **Ollama HTTP API, not the Python package:** same behaviour whether Ollama runs here or on another machine.
- **`OLLAMA_HOST_URL`** instead of `OLLAMA_HOST`, because Ollama itself uses `OLLAMA_HOST` in a different format.
- **Only ATLAS's own `.env`** is read (install folder or `~/.config/atlas/.env`), never your project's `.env`.

## Tests

```bash
.venv/bin/python -m pytest -q
```
Tests run real HTTP servers that imitate Ollama and OpenAI, so the actual provider, streaming, error handling, consent and CLI code is exercised without a GPU.

## Roadmap
- **Phase 2:** tools (`read_file`, `list_directory`, `search_files`, `write_file`, `edit_file`, `run_command`, `run_tests`, `git_status`, `git_diff`) with workspace boundaries, approval prompts and timeouts.
- **Phase 3:** tool-calling agent loop (plan → execute → observe → debug → verify).
- **Phase 4:** SQLite memory, repository index, hybrid routing.
- **Phase 5:** approved-example dataset, LoRA/QLoRA fine-tuning, evaluation.
