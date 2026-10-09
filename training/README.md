# Training JJ

Turn an open-source base model into your own JJ model with LoRA fine-tuning.

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

- **Python server:** `JJ_MODEL_ID=your-username/jj-1.5b uvicorn app:app --host 0.0.0.0 --port 8000`
- **Raspberry Pi with llama.cpp:** `llama-server -m jj-1.5b-q8_0.gguf --host 0.0.0.0 --port 8080`

## Making JJ stronger

1. **Better data first.** Write a few hundred high-quality conversations on what you want JJ to be good at.
2. **Bigger base.** Change `BASE_MODEL` to `Qwen/Qwen2.5-3B-Instruct` (needs more Colab memory and an 8 GB Pi).
3. **Preference tuning.** Later, collect pairs of good vs. bad answers and train with DPO (TRL's `DPOTrainer`).
4. **Evaluate.** Keep a list of test questions JJ never trained on and compare versions.
