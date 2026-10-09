# Training JJ

Turn an open-source base model into your own JJ model with LoRA fine-tuning.

Default base: **SmolLM2-360M-Instruct**, sized for a Raspberry Pi Zero 2 W (512 MB RAM).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joseph1-H/test/blob/main/training/train_jj.ipynb)

## Files

| File | What it is |
|---|---|
| `make_dataset.py` | The training conversations. Edit this to change who JJ is. |
| `jj_dataset.jsonl` | The generated dataset (run `python make_dataset.py` after editing). |
| `train_jj.ipynb` | Colab notebook: fine-tune, test, save, export to GGUF. |

## How to train

1. Tap **Open in Colab** above (works while the repo is public).
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**. Training takes about 5–15 minutes.
4. Check the test answers in step 5. JJ should say it's JJ.
5. Save the model: upload it to Hugging Face (6a) and/or download a GGUF for the Pi (6b).

## Using your trained model

- **Python server:** `JJ_MODEL_ID=your-username/jj-360m uvicorn app:app --host 0.0.0.0 --port 8000` (computers and Pi 4/5 only)
- **Raspberry Pi Zero with llama.cpp:** build llama.cpp on the Pi, then `llama-server -m jj-360m-Q4_K_M.gguf -c 1024 -t 4 --host 0.0.0.0 --port 8080`

## Making JJ stronger

1. **Better data first.** Write a few hundred high-quality conversations on what you want JJ to be good at.
2. **Bigger base.** A Pi Zero caps JJ at about 360M parameters. For a smarter JJ, move to a Pi 5 and use `Qwen/Qwen2.5-1.5B-Instruct` or bigger.
3. **Preference tuning.** Later, collect pairs of good vs. bad answers and train with DPO (TRL's `DPOTrainer`).
4. **Evaluate.** Keep a list of test questions JJ never trained on and compare versions.
