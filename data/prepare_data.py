"""
prepare_data.py — WikiText-2 loading and preprocessing sketch
for DSA4213 Assignment 1 (Parts I & II).

This produces a cleaned, concatenated text stream plus train/val/test
splits ready for tokenization. WikiText-2 shares the same validation/test
splits as WikiText-103 but has a much smaller train split (~2M tokens
vs. ~103M), so the full train split is already a computationally
reasonable size on its own — no subsetting needed by default. Adjust
SUBSET_FRACTION if you want an even smaller corpus.
"""

import re
from datasets import load_dataset

SUBSET_FRACTION = 1.0  # use the full WikiText-2 train split


def load_raw():
    """Load WikiText-2 (raw, unprocessed tokens) from the Hub."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    return ds  # DatasetDict with 'train', 'validation', 'test'


def clean_lines(lines):
    """
    WikiText rows include blank lines and section headers like
    '= Title =' or '= = Subheading = ='. Decide explicitly what to
    keep — this is a preprocessing decision you should document
    in the report.
    """
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  # drop blank lines
        # Drop section/subsection headers (lines wrapped in '=' signs)
        if re.match(r"^=+.*=+$", line):
            continue
        cleaned.append(line)
    return cleaned


def build_corpus(split_dataset, subset_fraction=1.0):
    """Concatenate a split's lines into one text stream, optionally subsetting."""
    lines = split_dataset["text"]
    if subset_fraction < 1.0:
        n = int(len(lines) * subset_fraction)
        lines = lines[:n]
    lines = clean_lines(lines)
    return "\n".join(lines)


def main():
    ds = load_raw()

    train_text = build_corpus(ds["train"], subset_fraction=SUBSET_FRACTION)
    val_text = build_corpus(ds["validation"], subset_fraction=1.0)
    test_text = build_corpus(ds["test"], subset_fraction=1.0)

    print(f"Train chars: {len(train_text):,}")
    print(f"Val chars:   {len(val_text):,}")
    print(f"Test chars:  {len(test_text):,}")

    # Save to data/processed/ for downstream tokenizer.py to consume
    with open("data/processed/train.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    with open("data/processed/valid.txt", "w", encoding="utf-8") as f:
        f.write(val_text)
    with open("data/processed/test.txt", "w", encoding="utf-8") as f:
        f.write(test_text)


if __name__ == "__main__":
    main()