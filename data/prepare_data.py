"""
prepare_data.py — WikiText-103 loading and preprocessing sketch
for DSA4213 Assignment 1 (Parts I & II).

This produces a cleaned, concatenated text stream plus train/val/test
splits ready for tokenization. Adjust SUBSET_FRACTION to keep the
project computationally reasonable.
"""

import re
from datasets import load_dataset

SUBSET_FRACTION = 0.05  # e.g. use ~5% of train for a manageable corpus size


def load_raw():
    """Load WikiText-103 (raw, unprocessed tokens) from the Hub."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
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