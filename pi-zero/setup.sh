#!/usr/bin/env bash
# JJ AI setup for Raspberry Pi Zero 2 W (also works on other Pis).
# Builds llama.cpp, installs a JJ model, and starts JJ as a service on port 8000.
#
# Usage (on the Pi):
#   bash setup.sh                     # uses models/jj.gguf if present, else downloads a test model
#   bash setup.sh /path/to/model.gguf # use a specific GGUF file (e.g. your trained JJ)
set -euo pipefail

JJ_DIR="$HOME/jj"
PORT=8000
THREADS=$(nproc)
# Test model used until you copy in your own trained JJ (models/jj.gguf).
TEST_MODEL_URL="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf"

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

ARCH=$(uname -m)
if [[ "$ARCH" == "armv6l" ]]; then
  echo "Warning: this looks like an original Pi Zero (ARMv6, 1 core). JJ will be very slow."
  echo "A Pi Zero 2 W with 64-bit Raspberry Pi OS Lite is strongly recommended."
  THREADS=1
fi

say "Installing build tools"
sudo apt-get update -y
sudo apt-get install -y git cmake build-essential curl

# 512 MB RAM is not enough to compile or run comfortably, so add 1 GB of swap.
if ! swapon --show | grep -q /swapfile-jj; then
  say "Adding 1 GB swap file"
  sudo fallocate -l 1G /swapfile-jj || sudo dd if=/dev/zero of=/swapfile-jj bs=1M count=1024
  sudo chmod 600 /swapfile-jj
  sudo mkswap /swapfile-jj
  sudo swapon /swapfile-jj
  grep -q /swapfile-jj /etc/fstab || echo '/swapfile-jj none swap sw 0 0' | sudo tee -a /etc/fstab
fi

mkdir -p "$JJ_DIR/models"
cd "$JJ_DIR"

if [[ ! -x llama.cpp/build/bin/llama-server ]]; then
  say "Building llama.cpp (this takes 30-60 minutes on a Pi Zero 2 W)"
  [[ -d llama.cpp ]] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
  cmake --build llama.cpp/build --target llama-server -j 2
fi

MODEL="${1:-$JJ_DIR/models/jj.gguf}"
if [[ ! -f "$MODEL" ]]; then
  say "No trained JJ model found; downloading a test model (~390 MB)"
  MODEL="$JJ_DIR/models/test-smollm2-360m.gguf"
  [[ -f "$MODEL" ]] || curl -fL -o "$MODEL" "$TEST_MODEL_URL" || {
    echo "Download failed. Copy a .gguf file to $JJ_DIR/models/jj.gguf and run this script again."
    exit 1
  }
fi
MODEL=$(realpath "$MODEL")

say "Installing the JJ service (model: $(basename "$MODEL"))"
sudo tee /etc/systemd/system/jj-ai.service > /dev/null <<EOF
[Unit]
Description=JJ AI (llama.cpp)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$JJ_DIR
ExecStart=$JJ_DIR/llama.cpp/build/bin/llama-server -m $MODEL -c 1024 -t $THREADS --host 0.0.0.0 --port $PORT
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable jj-ai
sudo systemctl restart jj-ai

say "Done! JJ is starting on port $PORT (loading the model takes a minute)."
echo "Test on the Pi:  curl http://localhost:$PORT/health"
echo "In the app, set the server URL to:  http://$(hostname).local:$PORT"
echo "Logs:  journalctl -u jj-ai -f"
