# DSA4213 Assignment 1

Code for Parts I–III of the DSA4213 individual assignment: language-model comparison (Part I), embedding ablation (Part II) and downstream classification (Part III).

> Items marked **TODO** still need details that aren't in this repo.

## Environment

- Python 3.12 (Google Colab runtime 2026.07)
- Hardware used for reported numbers: Google Colab **NVIDIA Tesla T4** (16 GB) with mixed precision (AMP). The n-gram model and Skip-gram training run on CPU (2 cores).
- Dependencies (pinned): `pip install -r requirements.txt`. These are torch 2.5.1, numpy 2.0.2, scipy 1.14.1, gensim 4.4.0, scikit-learn 1.5.2, datasets 3.1.0, pandas 2.2.2 and matplotlib 3.9.2. On Colab, only gensim needs installing; Part II Cell 1b does it.

No API keys, Kaggle tokens or Hugging Face tokens are needed or included. Colab may print an "HF_TOKEN not found" warning, which you can ignore.

## Data access (no data files are committed)

| Data | Used in | How to get it |
|---|---|---|
| WikiText-2 (raw), CC BY-SA 3.0 | Parts I, II | Hugging Face [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), config `wikitext-2-raw-v1`. Run `python data/prepare_data.py`, or let Cell 1 of the Part I notebook download it. Both write `data/processed/{train,valid,test}.txt`. |
| GloVe 6B, 100-d | Part II | `gensim.downloader.load("glove-wiki-gigaword-100")` in Part II Cell 5. This downloads about 128 MB to `~/gensim-data` the first time it runs. It is not bundled. |
| Rotten Tomatoes critic reviews | Part III | Kaggle: **TODO: dataset URL** (the `review_type` = Fresh/Rotten column suggests [Rotten Tomatoes movies and critic reviews](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)). Steps: (1) sign in to Kaggle and download the reviews CSV from the dataset page, or run `kaggle datasets download -d <owner>/<dataset>` with your own `~/.kaggle/kaggle.json`; (2) put it in `data/raw/`; (3) build `data/prepared/rt_{train,valid,test}.csv` (see Part III preprocessing). The CSVs are not redistributed here, and `.gitignore` excludes `data/raw/`, `data/prepared/` and `kaggle.json`. |

### Preprocessing (Parts I and II)

1. Use the official WikiText-2 train/validation/test splits as-is.
2. Undo WikiText escaping: ` @-@ ` → `-`, ` @,@ ` → `,`, ` @.@ ` → `.`.
3. Lowercase and normalise `’` to `'`.
4. Tokenize with the regex `\w+(?:'\w+)?|[^\w\s]` (words, with internal apostrophes kept, and single punctuation marks).
5. Wrap each non-empty line as `<s> … </s>` and concatenate everything into one token stream per split.
6. Vocabulary: the 30,000 most frequent **training** tokens, plus `<pad>` and `<unk>` (30,002 entries). Everything else maps to `<unk>`, which covers 2.13% of training tokens.
7. Neural LMs read non-overlapping windows of 32 tokens, with targets shifted by one.

Resulting sizes: 2,124,061 train / 221,501 valid / 249,435 test tokens.

### Preprocessing (Part III)

1. **TODO:** the step that builds `data/prepared/rt_{train,valid,test}.csv` from the Kaggle file isn't in the repo. It needs to be documented: the source columns, the filtering, class balancing and split seed. The prepared files have columns `text` and `label` (1 = Fresh, 0 = Rotten). They hold 20,000 train / 2,500 val / 2,500 test reviews, each split 50% positive.
2. Reviews are tokenized with the Part I tokenizer and vocabulary (loaded from `results/lstm.pt`) and encoded as `<s>` + tokens, with no `</s>`. They're truncated to 32 tokens and right-padded. Reviews with no tokens are dropped (none were). OOV rate is about 8.5%, and 25% of reviews are truncated.

## How to run

Run the notebooks in order in the same Google Drive folder. Each one mounts Drive and uses `/content/drive/MyDrive/dsa4213/ind_assignment` as its working directory; change `PROJECT_DIR` in Cell 0/1 if yours differs. Choose a T4 runtime, then **Run all**.

| Step | Notebook | Needs |
|---|---|---|
| Part I | `Part1_models_Colab.ipynb` | internet (downloads WikiText-2) |
| Part II | `Part2_embeddings_Colab.ipynb` | `data/processed/` from Part I; internet (GloVe) |
| Part III | `Part3_downstream_Colab.ipynb` | `data/prepared/`, `results/lstm.pt` (Part I), `results/part2/lstm_*.pt` (Part II) |

Part II is resumable: finished (setting, seed) runs are skipped and an interrupted run continues from its last epoch, so after a Colab disconnect you just re-run Cell 8.

### Where each table and figure comes from

| Report item | Notebook → cell | File (under `results/`) |
|---|---|---|
| n-gram add-k tuning + test PPL | Part I → Cell 4 | — |
| Part I perplexity / cost table | Part I → Cell 10 | `metrics_part1.csv` |
| Part I loss curves | Part I → Cell 11 | `loss_curves_part1.png` |
| Part I generations | Part I → Cell 12 | `generations.md` |
| Skip-gram training cost + nearest neighbours | Part II → Cell 4 | `part2/own_skipgram_cost.json` |
| GloVe vocabulary coverage | Part II → Cell 5 | `part2/coverage_glove-wiki-gigaword-100.json` |
| Part II per-seed, summary and paired tables | Part II → Cell 9 | `part2/runs_part2.csv`, `part2/summary_part2.csv`, `part2/paired_part2.csv` |
| Part II validation curves + test PPL figure | Part II → Cell 10 | `part2/valid_curves_part2.png` |
| Part II generations | Part II → Cell 11 | `part2/generations_part2.md` |
| Part III tokenization stats (OOV, truncation) | Part III → Cell 2 | `part3/tokenization_stats.csv` |
| Part III downstream table, LM-vs-downstream ranking, TF-IDF reference | Part III → Cell 5 | `part3/results_embeddings_summary.csv`, `part3/results_embeddings_per_seed.csv` |
| Part III error analysis + McNemar test | Part III → Cell 6 | `part3/errors_by_tag_*.csv`, `part3/error_table_*.csv` |

Checkpoints, Skip-gram models and feature matrices (`*.pt`, `*.model`, `*.npy`) are not committed.

### Expected runtime (Colab T4)

| Step | Time |
|---|---|
| Part I: n-gram fit | 9 s (CPU); dev-set tuning adds a few minutes (not timed) |
| Part I: RNN / LSTM / Transformer | 159 s / 252 s / 270 s (≈ 11 min total) |
| Part II: Skip-gram | 63 s on 2 CPU cores |
| Part II: GloVe download + load | 82 s |
| Part II: 12 LSTM runs (4 settings × 3 seeds, 20 epochs, ~21 s/epoch) | 406–440 s each, ≈ 85 min total |
| Part III | not recorded; feature extraction and logistic regression take minutes, not hours |

CPU-only runtimes weren't measured. Neural training is much slower on CPU.

## Configurations

### Part I — n-gram
- Trigram (orders 1–3) over the same token stream and 30k vocabulary as the neural models.
- Linear interpolation with fixed weights λ₁ = λ₂ = λ₃ = 1/3.
- Add-k grid {0.001, 0.01, 0.05, 0.1, 0.5, 1.0}, tuned on the validation set. Best k = 0.001.

### Part I — neural LMs (shared `BASE_CFG`, Cell 6)

| Setting | Value |
|---|---|
| Optimizer | Adam, LR 1e-3 |
| Batch size | 64 |
| Max epochs | 15, early stopping with patience 2 on validation PPL (best checkpoint restored) |
| Gradient clipping | max-norm 1.0 |
| Context window | 32 tokens |
| Embedding dim | 128 |
| Hidden dim | 256 (RNN/LSTM hidden size; Transformer feed-forward size) |
| RNN / LSTM | 1 layer, no dropout |
| Transformer | 2 layers, 4 heads, dropout 0.1, sinusoidal positions, causal mask |
| Loss | cross-entropy, reported in bits/token; PPL = 2^CE |
| Precision | fp16 autocast + GradScaler on GPU |

### Generation (Part I Cell 12, Part II Cell 11)
- Temperature 0.8, top-k 40, at most 30 new tokens, context 32 tokens, decoding seed 123. `<pad>` is never sampled, and generation stops at `</s>`.
- Five prompts, the same for every model within a part. Part II's prompts are slightly different wordings of Part I's.

### Part II — embedding ablation (LSTM)

| Setting | Embedding init | Trainable? |
|---|---|---|
| `random` | PyTorch default | yes |
| `own_sg` | own Skip-gram, trained on the WikiText-2 **train split only** | frozen |
| `public` | GloVe `glove-wiki-gigaword-100` | frozen |
| `public_ft` (extra) | GloVe `glove-wiki-gigaword-100` | yes (fine-tuned) |

- Embedding dim 100, to match GloVe. The `random` baseline is retrained in Part II with the same architecture as the other settings; it is not the Part I LSTM.
- LSTM (`LSTM_CFG`, Cell 7): 1 layer, hidden 256, dropout 0.3 on embeddings and outputs, projection 256→100 before an untied output layer, `padding_idx` = `<pad>`. Adam LR 1e-3, batch 64, window 32, max 20 epochs, early-stopping patience 2, gradient clipping 1.0, AMP. Only trainable parameters go to the optimizer.
- Skip-gram (`SG_CFG`, Cell 2): gensim `Word2Vec`, `sg=1`, vector_size 100, window 5, negative 5, min_count 1, epochs 5, workers 1, seed 42. It's trained on the `<unk>`-mapped training stream, so every vocabulary entry gets a vector.
- GloVe coverage: 96.39% of vocabulary types (28,914 / 29,998) and 99.66% of in-vocabulary training tokens.
- OOV policy (Cell 6): `<pad>` → zeros; `<unk>` → mean of the found vectors; other missing words (including `<s>` and `</s>` for GloVe) → N(0, σ²), where σ is the std of the found vectors (NumPy RNG seed 0).
- Stability: 3 seeds per setting. The seed fixes LSTM init and batch order, so settings are also compared paired by seed.

### Part III — downstream sentiment classification
- Task: binary sentiment on Rotten Tomatoes critic reviews (Fresh = 1, Rotten = 0).
- Representations: frozen LM hidden states (256-d) from the Part II LSTMs (all 4 settings × 3 seeds) and the Part I LSTM. Two kinds of pooling are used: **last** (state at the final real token) and **mean** (average over real tokens, including `<s>`, excluding padding). The main table uses mean pooling for `random` / `own_sg` / `public`.
- Classifier: `StandardScaler` + scikit-learn `LogisticRegression` (`max_iter=3000`). C is chosen from {1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100} by validation macro-F1, then evaluated once on the test set. Metrics are accuracy and macro-F1.
- References: majority class, and TF-IDF (1–2-grams, `min_df=2`, same tokenizer) + logistic regression with the same C search.
- Error analysis: error rates on tagged subsets (negation, contrast, truncated, OOV-heavy, very short), plus McNemar's exact test between `random` and `public` for each LM seed.

## Random seeds

| What | Seed |
|---|---|
| Part I neural LM training (init + shuffling) | 42 |
| Part II LM runs (init + batch order) | 42, 43, 44 (mean ± std reported) |
| Part II Skip-gram | 42 (`workers=1` keeps it deterministic) |
| Part II OOV vector initialisation | 0 |
| Generation / decoding (Parts I, II) | 123 |
| Part III global seed (`set_seed`) | 42; features come from the Part II seed-42/43/44 LMs |
| Rotten Tomatoes train/val/test split | **TODO** |

`set_seed()` seeds `random`, `numpy`, `torch` and CUDA. GPU kernels are not forced to be deterministic, so re-runs on a T4 can differ slightly in the last decimal places.

## Repository layout

```
Part1_models_Colab.ipynb       Part I   (outputs kept so results are visible)
Part2_embeddings_Colab.ipynb   Part II  (outputs kept)
Part3_downstream_Colab.ipynb   Part III (outputs kept)
data/prepare_data.py           WikiText-2 download → data/processed/
src/evaluate.py                CE / PPL helper used by src/part1_models/RNN.py
src/part1_models/              early standalone scripts and a superseded local draft notebook;
                               the reported results come from the notebooks above
requirements.txt
```

## Attribution

- The n-gram LM, tokenizer and evaluation code are adapted from the DSA4213 Lecture 1 notebook. The RNN, LSTM and Transformer LMs are adapted from the DSA4213 Lecture 3 notebook and reused in Parts II–III. Each adapted file or notebook says so at the top.
- Datasets: WikiText-2 (Merity et al., 2016; CC BY-SA 3.0); Rotten Tomatoes critic reviews (Kaggle, see above). Pretrained embeddings: GloVe 6B (Pennington et al., 2014), via gensim-data. Libraries: PyTorch, gensim, scikit-learn, Hugging Face `datasets`.
- AI coding assistance was used; see the report for details.
