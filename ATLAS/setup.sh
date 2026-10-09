#!/usr/bin/env bash
# ATLAS Phase 1 setup for Linux (tested target: Linux Mint 22.x, NVIDIA GPU).
# Run from the ATLAS folder:  bash setup.sh
# Every step checks its own result. Nothing is downloaded or installed system-wide without asking.
set -uo pipefail

cd "$(dirname "$0")"
ROOT=$(pwd)
OLLAMA_URL="http://localhost:11434"
FAILS=0

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✔\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✘\033[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }
ask()  { local a; [[ -t 0 ]] || return 1; read -r -p "  $1 [y/N] " a; [[ "$a" =~ ^[Yy] ]]; }

# ---------------------------------------------------------------------------
say "1/7 Python"
if ! command -v python3 >/dev/null; then
  bad "python3 not found. Install it: sudo apt install python3"; exit 1
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  bad "Python 3.10+ required, found $(python3 --version)"; exit 1
fi
ok "$(python3 --version)"
if ! python3 -c 'import venv, ensurepip' 2>/dev/null; then
  bad "python3-venv is missing. Install it: sudo apt install python3-venv"; exit 1
fi

# ---------------------------------------------------------------------------
say "2/7 Python environment (.venv)"
[[ -d .venv ]] || python3 -m venv .venv || { bad "could not create .venv"; exit 1; }
.venv/bin/python -m pip install -q --upgrade pip
if .venv/bin/python -m pip install -q -e ".[dev]"; then
  ok "installed ATLAS and dependencies into .venv"
else
  bad "pip install failed (see above)"; exit 1
fi
.venv/bin/atlas --version >/dev/null && ok "$(.venv/bin/atlas --version)"

# ---------------------------------------------------------------------------
say "3/7 Configuration (.env)"
if [[ -f .env ]]; then
  ok ".env already exists (left unchanged)"
else
  cp .env.example .env && ok "created .env from .env.example"
fi
chmod 600 .env && ok ".env permissions set to owner-only (600)"
MODEL=$(grep -E '^ATLAS_LOCAL_MODEL=' .env | tail -1 | cut -d= -f2-)
MODEL=${MODEL:-qwen2.5-coder:7b}

# ---------------------------------------------------------------------------
say "4/7 NVIDIA GPU"
if command -v nvidia-smi >/dev/null && GPU=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null) && [[ -n "$GPU" ]]; then
  ok "$GPU"
else
  warn "NVIDIA driver not working. Install it from Linux Mint's Driver Manager, then reboot."
  warn "ATLAS still works, but the model will run on the CPU (much slower)."
fi

# ---------------------------------------------------------------------------
say "5/7 Ollama"
if ! command -v ollama >/dev/null; then
  warn "Ollama is not installed."
  if ask "Install Ollama now with the official script (https://ollama.com/install.sh)?"; then
    curl -fsSL https://ollama.com/install.sh | sh || bad "Ollama install failed"
  fi
fi
if command -v ollama >/dev/null; then
  ok "$(ollama --version 2>/dev/null | head -1)"
  if ! curl -fsS "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
    warn "Ollama server not running; trying to start it"
    sudo systemctl start ollama 2>/dev/null || (nohup ollama serve >/dev/null 2>&1 &)
    sleep 3
  fi
  if curl -fsS "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
    ok "Ollama server reachable at $OLLAMA_URL"
  else
    bad "Ollama server not reachable at $OLLAMA_URL. Start it with: ollama serve"
  fi
else
  bad "Ollama not installed. Get it from https://ollama.com/download"
fi

# ---------------------------------------------------------------------------
say "6/7 Local model ($MODEL)"
if command -v ollama >/dev/null && curl -fsS "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
  if ollama list | awk 'NR>1 {print $1}' | grep -qx -e "$MODEL" -e "$MODEL:latest"; then
    ok "$MODEL is downloaded"
  elif ask "Download $MODEL now? (qwen2.5-coder:7b is about 4.7 GB)"; then
    ollama pull "$MODEL" && ok "downloaded $MODEL" || bad "ollama pull $MODEL failed"
  else
    bad "$MODEL not downloaded. Run: ollama pull $MODEL"
  fi
else
  warn "skipped: Ollama not available"
fi

# ---------------------------------------------------------------------------
say "7/7 Tests"
if .venv/bin/python -m pytest -q; then ok "unit tests passed"; else bad "unit tests failed"; fi

echo
.venv/bin/atlas --workspace "$ROOT" doctor || true

if curl -fsS "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
  say "Real inference test (local model)"
  REPLY=$(timeout 300 .venv/bin/atlas --workspace "$ROOT" ask "Reply with exactly these two words and nothing else: ATLAS OK" 2>&1)
  echo "  model replied: $REPLY"
  if grep -qi "ATLAS OK" <<<"$REPLY"; then ok "local inference works"; else bad "unexpected reply from the model"; fi
fi

# ---------------------------------------------------------------------------
mkdir -p "$HOME/.local/bin"
if [[ ! -e "$HOME/.local/bin/atlas" ]] && ask "Add the 'atlas' command to ~/.local/bin so it works from any folder?"; then
  ln -s "$ROOT/.venv/bin/atlas" "$HOME/.local/bin/atlas" && ok "linked ~/.local/bin/atlas"
fi

echo
if (( FAILS == 0 )); then
  printf '\033[1;32mATLAS is ready.\033[0m Start it with:\n  %s/.venv/bin/atlas --workspace /path/to/project\n' "$ROOT"
else
  printf '\033[1;31m%d problem(s) found.\033[0m Fix the ✘ items above and run this script again.\n' "$FAILS"
  exit 1
fi
