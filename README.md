# DSA4213 Assignment 1

Code for the DSA4213 individual assignment:

- **Part I:** language-model comparison (trigram, RNN, LSTM, Transformer) on WikiText-2.
- **Part II:** embedding ablation on the LSTM (random / own Skip-gram / GloVe).
- **Part III:** downstream sentiment classification on Rotten Tomatoes critic reviews, using frozen LSTM representations.

> **Read first.** Part III uses Part II checkpoints from a second training run with dropout 0.3, while the Part II notebook shows the dropout 0.0 run. See [Known discrepancies](#known-discrepancies). The report explains this in its "Note on checkpoints" (Part III).

## Environment

- Python 3.12, Google Colab (runtime 2026.07).
- Hardware for all reported numbers: **NVIDIA Tesla T4** (16 GB), mixed precision (fp16 autocast + GradScaler). The n-gram model, Skip-gram and Rotten Tomatoes preprocessing run on CPU (2 cores).
- Dependencies: `pip install -r requirements.txt`. Pinned versions: torch 2.5.1, numpy 2.0.2, scipy 1.14.1, gensim 4.4.0, scikit-learn 1.5.2, datasets 3.1.0, pandas 2.2.2, matplotlib 3.9.2, kagglehub. On Colab, only gensim needs installing (Part II Cell 1b does it).
- No API keys or tokens are included. Colab may warn that `HF_TOKEN` is not set; this can be ignored, because WikiText-2 is public.

## Data access

Every dataset is downloaded from its original source by the notebooks.

| Data | Used in | How to get it |
|---|---|---|
| WikiText-2 raw (CC BY-SA 3.0) | Parts I–III | Hugging Face [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), config `wikitext-2-raw-v1`. Downloaded automatically by Part I Cell 1 or `python data/prepare_data.py` (every notebook calls this if `data/processed/` is missing). Writes `data/processed/{train,valid,test}.txt`. |
| GloVe 6B, 100-d | Part II | `gensim.downloader.load("glove-wiki-gigaword-100")` (Part II Cell 5), about 128 MB, cached in `~/gensim-data`. |
| Rotten Tomatoes critic reviews | Part III | Kaggle: [stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset). Downloaded automatically with `kagglehub` (Cell 0 of `Part3_data_prep_RottenTomatoes.ipynb`). If that fails, Colab asks you to upload `rotten_tomatoes_critic_reviews.csv` (or the Kaggle `archive.zip`); outside Colab, place the CSV in `data/raw/` yourself. |

`.gitignore` excludes `data/raw/`, `data/prepared/`, `kaggle.json`, and all checkpoint and feature files (`*.pt`, `*.model`, `*.npy`).

### Preprocessing: WikiText-2 (Parts I and II)

1. Use the official train / validation / test splits.
2. Undo WikiText escaping: ` @-@ ` → `-`, ` @,@ ` → `,`, ` @.@ ` → `.`.
3. Lowercase the text and normalise `’` to `'`.
4. Tokenize with the regex `\w+(?:'\w+)?|[^\w\s]`.
5. Wrap each non-empty line in `<s> … </s>` and concatenate everything into one token stream per split.
6. Build the vocabulary from the 30,000 most frequent **training** tokens, plus `<pad>` and `<unk>` (30,002 entries). Other tokens map to `<unk>` (2.13% of training tokens).
7. Neural LMs read non-overlapping 32-token windows, with targets shifted by one. No padding occurs, but `<pad>` is still excluded from the loss via `ignore_index`.

Sizes: 2,124,061 train / 221,501 valid / 249,435 test tokens.

### Preprocessing: Rotten Tomatoes (Part III)

`Part3_data_prep_RottenTomatoes.ipynb` (CPU, deterministic given `SEED = 42`):

1. **Load** 1,130,017 rows and rename `rotten_tomatoes_link` → `movie_id`, `review_content` → `text`, `review_type` → `label` (Fresh = 1, Rotten = 0).
2. **Clean:** drop rows with missing fields (1,064,211 remain), strip `[Full review in …]` tags, and collapse whitespace. Drop reviews with fewer than 4 tokens (Part I tokenizer). 1,058,330 rows remain.
3. **Deduplicate** on normalised text (lowercase alphanumerics): keep one copy of each repeated text, and drop every copy of texts whose copies have conflicting labels (700 rows). 944,430 rows remain.
4. **WikiText overlap filter:** drop a review if at least 50% of its 8-grams occur in WikiText-2 train, or if it shares any 13-gram with it. 497 reviews share at least one 8-gram; **64** Rotten Tomato Reviews meet this rule and are removed (listed in `data/prepared/overlap_removed.csv`).
5. **Split by movie:** 80 / 10 / 10% of movies go to train / valid / test (seed 42). No movie appears in two splits.
6. **Cap and balance:** keep at most 10 reviews per movie, then sample balanced Fresh/Rotten sets of 20,000 / 2,500 / 2,500 reviews from 10,831 / 1,347 / 1,342 movies (sampling seeds 42 / 43 / 44).
7. **Assert** that no movie and no normalised text is shared across splits, then write `data/prepared/rt_{train,valid,test}.csv` and `prep_stats.json`.

The overlap filter reads `data/processed/train.txt` from Part I (23,767 non-empty lines, 2,076,521 tokens without `<s>`/`</s>`). The committed run of this notebook reproduces the splits used for the reported results (64 reviews removed; `<unk>` rates 8.53 / 8.49 / 8.61%, identical to Part III Cell 2).

Part III then encodes each review as `<s>` + tokens (no `</s>`), using the Part I tokenizer and vocabulary loaded from `results/lstm.pt`. Reviews are truncated to 32 tokens and right-padded. About 8.5% of review tokens are `<unk>`, and 25% of reviews are truncated.

## How to run

Every notebook starts with the same **Cell 0**. On Colab it clones this repo to `/content/DSA4213_Assignment` and works there; locally it uses the repo root. Choose a T4 runtime, then **Run all**. Each notebook is self-contained: missing inputs are rebuilt or downloaded at the start, so the notebooks can run in separate Colab runtimes.

| Step | Notebook | What it fetches or builds automatically | Writes |
|---|---|---|---|
| 1 | `Part1_models_Colab.ipynb` | WikiText-2 | `results/` (Part I tables, curves, generations, `lstm.pt`) |
| 2 | `Part2_embeddings_Colab.ipynb` | WikiText-2, GloVe, gensim | `results/part2/` |
| 3 | `Part3_data_prep_RottenTomatoes.ipynb` | WikiText-2, Kaggle CSV (`kagglehub`) | `data/prepared/` |
| 4 | `Part3_downstream_Colab.ipynb` | Cell 0b runs step 3 if `data/prepared/` is missing, and downloads the trained checkpoints from the GitHub release | `results/part3/` |

**Part III without retraining.** The LSTM checkpoints used in Part III (`results/lstm.pt`, `results/part2/lstm_{setting}_s{seed}.pt` and `result_{setting}_s{seed}.json`) are attached to GitHub release **v1.0** as `checkpoints.zip`. `ensure_checkpoints()` in Part III Cell 0b downloads and unzips it, so Part III runs end to end without training any language model.

**Resuming Part II.** Finished (setting, seed) runs are skipped and an interrupted run continues from its last epoch, so after a Colab disconnect just re-run Cell 8.

## Known discrepancies

The training notebooks were not all re-run after their final edits, because of Colab compute limits. These cases are listed so every reported number can be traced to its source.

1. **Two Part II runs.**

   | | Part II notebook outputs (Cells 8–11; report Part II) | Checkpoints in release v1.0 (report Part III) |
   |---|---|---|
   | LSTM dropout | 0.0 | 0.3 |
   | Test PPL, random / own_sg / public (mean of 3 seeds) | 168.57 / 153.42 / 146.16 | 141.2 / 141.7 / 137.5 |
   | Early stopping | yes, after 9–14 epochs | no, all 9 runs trained the full 20 epochs (`epochs_run` in `result_*.json`) |

   Everything else is identical: architecture, vocabulary, data, Skip-gram and GloVe matrices, seeds 42/43/44, and training procedure. Part III features, the `lm_test_ppl` column in Part III Cell 5 and the LM-vs-downstream ranking all come from the **dropout 0.3 checkpoints**. To reproduce them, set `dropout=0.3` in `LSTM_CFG` (Part II Cell 7) before running Cell 8.

2. **Part II Cell 12 has no output.** It was added after the dropout 0.0 run and only packages the checkpoints into `checkpoints.zip`. The release zip was built from the dropout 0.3 run instead and holds exactly 19 files: `results/lstm.pt` plus `lstm_{setting}_s{seed}.pt` and `result_{setting}_s{seed}.json` for random / own_sg / public × seeds 42 / 43 / 44.

3. **Notebook text.** Part I's markdown says Cell 0 mounts Google Drive; it actually clones the repo, as described above.

## Configurations

### Part I: n-gram
- A trigram (orders 1–3) over the same token stream and vocabulary as the neural models. |V| = 30,001, excluding `<pad>`.
- Each order uses add-k smoothing, and the orders are combined by linear interpolation λ₁P₁ + λ₂P₂ + λ₃P₃.
- k and λ are tuned jointly on the validation set:
  - k ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2};
  - λ is searched on a simplex grid in steps of 0.05.
- Best: k = 3e-5, λ = (0.25, 0.55, 0.20). Validation PPL 259.73, **test PPL 246.90** (7.95 bits/token).
- Reported fit time (8.65 s) is the counting step only; the tuning grid is not timed. The size column counts stored n-gram entries (2,009,444), not parameters.

### Part I: neural LMs (`BASE_CFG`, Cell 6)

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-3, batch 64, gradient clipping 1.0 |
| Budget | 20 epochs max, early-stopping patience 2 on validation PPL. All three models used the full 20 epochs, with validation loss still falling. |
| Context | 32-token windows; the hidden state is not carried across windows |
| Dimensions | embedding 100, hidden 256 (Transformer: feed-forward 256) |
| Dropout | 0.3 on embeddings and outputs (Transformer: also inside its layers) |
| Output layer | tied to the input embedding (RNN/LSTM project 256 → 100 first) |
| RNN / LSTM | 1 layer |
| Transformer | decoder-only, causal mask, 2 layers, 4 heads, post-LN, sinusoidal positions |
| Parameters | RNN 3.15M, LSTM 3.42M, Transformer 3.21M |
| Loss | cross-entropy in bits/token, PPL = 2^CE |
| Timing | training time = sum of epoch times, synchronised, excluding validation. Inference throughput = tokens/s over the test set (batch 64, fp16, T4). The n-gram's throughput is measured in pure Python on CPU, so it is not hardware-comparable. |

### Generation (Part I Cell 12, Part II Cell 11)
- Temperature 0.8, top-k 40, at most 30 new tokens, 32-token context, decoding seed 123.
- `<pad>` is never sampled. Generation stops early if `</s>` is sampled, so lengths can differ.
- The same five prompts are used in both parts. Part I also samples from the trigram with the same settings.

### Part II: embedding ablation (LSTM)

| Setting | Input embedding | Trainable? |
|---|---|---|
| `random` | PyTorch default init | yes |
| `own_sg` | own Skip-gram, trained on the WikiText-2 **train split only** | frozen |
| `public` | GloVe `glove-wiki-gigaword-100` | frozen |

- **LSTM (`LSTM_CFG`, Cell 7):** the same as the Part I LSTM, except for:
  - an **untied** output layer, so freezing the input table never freezes the output layer (this is also why the Part I LSTM cannot be reused as the `random` setting);
  - `padding_idx` = `<pad>`;
  - dropout 0.0 in the committed notebook (see [Known discrepancies](#known-discrepancies)).

  Embedding dimension 100 (matches GloVe), hidden 256, Adam 1e-3, batch 64, window 32, max 20 epochs, patience 2, clipping 1.0, AMP. Only trainable parameters go to the optimizer: 6.42M for `random`, 3.42M for the frozen settings.
- **Skip-gram (`SG_CFG`, Cell 2):** gensim `Word2Vec`, `sg=1`, vector size 100, window 5, negative 5, min_count 1, 5 epochs, `workers=1`, seed 42. It is trained on the `<unk>`-mapped training stream. **Cost: 60.1 s on 2 CPU cores**, reported separately from LM training.
- **GloVe coverage:** 96.39% of vocabulary types (28,914 / 29,998), and 99.66% of in-vocabulary training tokens.
- **OOV initialisation:**
  - `<pad>` → zeros;
  - `<unk>` → mean of the found vectors;
  - other missing words, including `<s>`/`</s>` for GloVe → N(0, σ²), where σ is the std of the found vectors (NumPy seed 0).

  The whole table is frozen.
- **Stability:** 3 seeds per setting (42, 43, 44). A seed fixes both LSTM initialisation and batch order, so settings are also compared paired by seed.

### Part III: downstream sentiment classification
- **Task:** binary sentiment (Fresh / Rotten), 20,000 / 2,500 / 2,500 balanced reviews.
- **Representations:** frozen 256-d LSTM hidden states from the Part II LSTMs (3 settings × 3 seeds). **Mean pooling** averages over real tokens (including `<s>`, excluding padding) and is used in the main table. **Last pooling** takes the state at the final real token; it is compared with mean pooling only in the validation check (Cell 4, C = 1), where mean pooling wins for every setting.
  - Cell 3 checks that the hidden states reproduce the model's logits and that padding does not change them. It also extracts Part I LSTM features as a sanity check; these are not in the report.
- **Classifier:** `StandardScaler` + `LogisticRegression` (`max_iter=3000`) for every representation. C is chosen from {1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100} by validation macro-F1, then the test set is evaluated **once**. Metrics: accuracy and macro-F1.
- **References:** majority class (0.500), and TF-IDF 1–2-grams (`min_df=2`, same tokenizer) + logistic regression with the same C search (accuracy 0.759).
- **Error analysis (Cell 6):** error rates on tagged subsets (negation, contrast, truncated = at least 32 review tokens, `<unk>`-heavy = at least 15% `<unk>`, very short = at most 6 tokens), plus McNemar's exact test between `random` and `public` for each LM seed.

## Where each result comes from

| Report item | Notebook → cell | Output file |
|---|---|---|
| n-gram tuning + test PPL | Part I → 4 | `results/ngram_tuning.json`, `result_trigram.json` |
| Part I quality / cost table | Part I → 10 | `results/metrics_part1.csv` |
| Part I loss curves | Part I → 11 | `results/loss_curves_part1.png` |
| Part I generations | Part I → 12 | `results/generations.md` |
| Skip-gram cost + nearest neighbours | Part II → 4 | `results/part2/own_skipgram_cost.json` |
| GloVe coverage | Part II → 5 | `results/part2/coverage_glove-wiki-gigaword-100.json` |
| Part II tables | Part II → 9 | `results/part2/runs_part2.csv`, `summary_part2.csv`, `paired_part2.csv` |
| Part II curves + test-PPL figure | Part II → 10 | `results/part2/valid_curves_part2.png` |
| Part II generations | Part II → 11 | `results/part2/generations_part2.md` |
| RT cleaning / split statistics | RT prep → 3–7 | `data/prepared/prep_stats.json`, `overlap_removed.csv` |
| Part III `<unk>` rate / truncation | Part III → 2 | `results/part3/tokenization_stats.csv` |
| Last vs mean pooling (validation) | Part III → 4 | printed in the notebook |
| Part III main table, ranking, TF-IDF | Part III → 5 | `results/part3/results_embeddings_summary.csv`, `results_embeddings_per_seed.csv` |
| Part III LM test PPL (dropout 0.3 checkpoints) | release v1.0 | `results/part2/result_{setting}_s{seed}.json` |
| Part III error analysis + McNemar | Part III → 6 | `results/part3/errors_by_tag_*.csv`, `error_table_*.csv` |

## Runtime (Colab T4, from the committed notebook outputs)

| Step | Time |
|---|---|
| Part I trigram fit | 8.65 s (CPU), plus a few minutes of untimed tuning |
| Part I RNN / LSTM / Transformer (20 epochs each) | 404 s / 427 s / 495 s |
| Part II Skip-gram | 60.1 s (2 CPU cores) |
| Part II GloVe download + load | about 65 s |
| Part II one LSTM run (dropout 0.0) | 197 ± 13 s (random), 264 ± 23 s (own_sg), 258 ± 20 s (public) |
| Rotten Tomatoes preprocessing | a few minutes on CPU (not timed) |
| Part III extraction + classifiers | a few minutes (not timed) |

## Random seeds

| What | Seed |
|---|---|
| Part I neural LMs (init + shuffling) | 42 |
| Part II LM runs (init + batch order) | 42, 43, 44 |
| Skip-gram | 42 (`workers=1`) |
| OOV vector initialisation | 0 |
| Generation | 123 |
| Rotten Tomatoes movie split / per-split sampling | 42 / 42, 43, 44 |
| Part III global seed (`set_seed`) | 42 |

`set_seed()` seeds `random`, `numpy`, `torch` and CUDA. GPU kernels are not forced to be deterministic, so re-runs can differ slightly in the last decimals.

## Repository layout

```
Part1_models_Colab.ipynb                Part I (outputs kept)
Part2_embeddings_Colab.ipynb            Part II (outputs kept; see Known discrepancies)
Part3_data_prep_RottenTomatoes.ipynb    Rotten Tomatoes cleaning + splits (outputs kept)
Part3_downstream_Colab.ipynb            Part III (outputs kept)
data/prepare_data.py                    WikiText-2 download -> data/processed/
requirements.txt
README.md
```

## Attribution

- The tokenizer, n-gram LM and evaluation code are adapted from the DSA4213 Lecture 1 notebook. The RNN and LSTM LMs are adapted from the DSA4213 Lecture 3 notebook and reused in Parts II–III.
- Datasets: WikiText-2 (Merity et al., 2016; CC BY-SA 3.0) and Rotten Tomatoes critic reviews (Kaggle, linked above, not redistributed).
- Pretrained embeddings: GloVe 6B (Pennington et al., 2014), via gensim-data.
- Libraries: PyTorch, gensim, scikit-learn, scipy, Hugging Face `datasets`, kagglehub, pandas, matplotlib.
- AI coding assistance was used; the report describes how.
