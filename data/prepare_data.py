"""
prepare_data.py — download WikiText-2 (raw) for DSA4213 Assignment 1 (Parts I & II).

Writes data/processed/{train,valid,test}.txt exactly as Part1_models_Colab.ipynb
(Cell 1) does: each split's raw lines concatenated unchanged. All cleaning
(@-@/@,@/@.@ markers, lowercasing, tokenizing, <s>/</s> per line, 30k vocab)
happens at tokenization time in the notebook, so no text is altered here.

Source: https://huggingface.co/datasets/Salesforce/wikitext (CC BY-SA 3.0).
Public dataset — no Hugging Face token needed.

Run from the repo root:  python data/prepare_data.py
"""

from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("data/processed")


def main():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "train"), ("validation", "valid"), ("test", "test")):
        path = OUT_DIR / f"{name}.txt"
        path.write_text("".join(ds[split]["text"]), encoding="utf-8")
        print(f"{path}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
