# JJ AI

Your own AI assistant, powered by an open-source Hugging Face model and served from your own Raspberry Pi.

```
┌──────────────┐   HTTP (Wi-Fi)   ┌──────────────────────────────┐
│ Flutter app  │ ───────────────▶ │ Raspberry Pi                 │
│ (iPhone etc.)│ ◀─────────────── │ FastAPI + Hugging Face model │
└──────────────┘     /chat        └──────────────────────────────┘
```

- `server/` – Python server that loads the model and exposes a `/chat` API.
- `app/` – Flutter chat app (iOS, Android, web) that talks to the server.

## Model

Default: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache 2.0). It's small enough to run on a Raspberry Pi CPU. Switch models with the `JJ_MODEL_ID` environment variable, e.g. `Qwen/Qwen2.5-1.5B-Instruct` for better answers if your Pi has 8 GB RAM.

## 1. Run the server

On any computer first, then on the Pi (Pi 5 or Pi 4 with 4–8 GB RAM, 64-bit Raspberry Pi OS):

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

The first start downloads the model (~1 GB). Test it:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi JJ!"}]}'
```

### Settings (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `JJ_MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model to load |
| `JJ_SYSTEM_PROMPT` | "You are JJ…" | JJ's personality |
| `JJ_MAX_NEW_TOKENS` | `256` | Max reply length |
| `JJ_API_KEY` | *(empty)* | If set, clients must send this key |
| `JJ_THREADS` | CPU count | CPU threads for inference |

### Start automatically on the Pi

Clone the repo to `/home/pi/JJ-AI`, set up the venv as above, then:

```bash
sudo cp server/jj-ai.service /etc/systemd/system/
sudo systemctl enable --now jj-ai
```

## 2. Run the app

```bash
cd app
flutter create . --project-name jj_ai --platforms ios,android,web
flutter pub get
flutter run
```

Tap ⚙️ in the app to set the server URL (default `http://raspberrypi.local:8000`).

**iOS note:** iOS blocks plain `http://` by default. Add this to `app/ios/Runner/Info.plist` inside the top `<dict>` so the app can reach the Pi on your home network:

```xml
<key>NSAppTransportSecurity</key>
<dict>
  <key>NSAllowsLocalNetworking</key>
  <true/>
</dict>
<key>NSLocalNetworkUsageDescription</key>
<string>JJ AI connects to your server on the local network.</string>
```

## Roadmap

- [ ] Streaming replies (word by word)
- [ ] Faster Pi inference with a quantized GGUF model via llama.cpp
- [ ] Fine-tune the model on JJ's own data (LoRA)
- [ ] Access from outside the home network (e.g. Tailscale)
