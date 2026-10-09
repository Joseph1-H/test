# JJ on a Raspberry Pi Zero

The Pi Zero 2 W has 512 MB of RAM, so JJ runs with [llama.cpp](https://github.com/ggml-org/llama.cpp) and a small 4-bit model instead of the Python server.

## What you need
- Raspberry Pi Zero 2 W (an original Pi Zero works but is very slow)
- microSD card, 16 GB or more
- 5V 2.5A power supply
- Raspberry Pi OS **Lite (64-bit)**

## 1. Prepare the SD card
Use **Raspberry Pi Imager** on a computer:
1. Choose **Raspberry Pi Zero 2 W** → **Raspberry Pi OS Lite (64-bit)**.
2. In the settings (⚙️ / "Edit settings"): set hostname `raspberrypi`, a username and password, your **Wi-Fi**, and turn on **SSH**.
3. Write the card, put it in the Pi, power it on, and wait 2 minutes.

## 2. Install JJ
From a computer on the same Wi-Fi:
```bash
ssh your-username@raspberrypi.local
git clone https://github.com/Joseph1-H/test.git jj-repo
bash jj-repo/pi-zero/setup.sh
```
This builds llama.cpp (30–60 minutes the first time), downloads a test model, and starts JJ automatically on every boot.

## 3. Use your trained JJ
After training in Colab (step 6b of `training/train_jj.ipynb`), copy the GGUF file to the Pi and rerun setup:
```bash
scp jj-360m-Q4_K_M.gguf your-username@raspberrypi.local:~/jj/models/jj.gguf
ssh your-username@raspberrypi.local 'bash jj-repo/pi-zero/setup.sh'
```

## 4. Chat
- **App:** set the server URL to `http://raspberrypi.local:8000`.
- **Browser:** open `http://raspberrypi.local:8000` for llama.cpp's built-in chat page.

Expect roughly a few words per second. Keep replies short for the best experience.

## Troubleshooting
- `journalctl -u jj-ai -f` shows the logs.
- If JJ crashes when loading, the model is too big: use a 135M model or a smaller quantization.
