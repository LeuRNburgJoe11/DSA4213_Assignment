# DSA4213 Assignment 1

Code for Parts I–III of the DSA4213 individual assignment: language-model comparison (Part I), embedding ablation (Part II) and downstream classification (Part III).

> Items marked **TODO** are Part II/III details that aren't in this repo yet. Fill them in when those notebooks are added.

## Environment

- Python 3.12 (Google Colab runtime)
- Hardware used for reported numbers: Google Colab **NVIDIA Tesla T4** (16 GB) with mixed precision (AMP). The n-gram model runs on CPU.
- Dependencies (pinned): `pip install -r requirements.txt`. These are torch 2.5.1, numpy 1.26.4, scipy 1.13.1, gensim 4.3.3, scikit-learn 1.5.2, datasets 3.1.0, pandas 2.2.2 and matplotlib 3.9.2.

No API keys, Kaggle tokens or Hugging Face tokens are needed or included. Colab may print an "HF_TOKEN not found" warning, which you can ignore.

## Data access (no data files are committed)

| Data | Used in | How to get it |
|---|---|---|
| WikiText-2 (raw), CC BY-SA 3.0 | Parts I, II | Hugging Face [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), config `wikitext-2-raw-v1`. Run `python data/prepare_data.py`, or let Cell 1 of the Part I notebook download it. Both write `data/processed/{train,valid,test}.txt`. |
| GloVe 6B, 100-d | Part II | `gensim.downloader.load("glove-wiki-gigaword-100")`. This downloads about 128 MB to `~/gensim-data` the first time it runs. It is not bundled. |
| Rotten Tomatoes reviews | Part III | Kaggle: **TODO: dataset URL**. Steps: (1) sign in to Kaggle and accept the dataset terms; (2) download the CSV from the dataset page, or run `kaggle datasets download -d <owner>/<dataset>` with your own `~/.kaggle/kaggle.json`; (3) put it at `data/raw/<file>.csv`. The CSV is not redistributed here, and `.gitignore` excludes `data/raw/`, `*.csv` and `kaggle.json`. |

### Preprocessing (Parts I and II)

1. Use the official WikiText-2 train/validation/test splits as-is.
2. Undo WikiText escaping: ` @-@ ` → `-`, ` @,@ ` → `,`, ` @.@ ` → `.`.
3. Lowercase and normalise `’` to `'`.
4. Tokenize with the regex `\w+(?:'\w+)?|[^\w\s]` (words, with internal apostrophes kept, and single punctuation marks).
5. Wrap each non-empty line as `<s> … </s>` and concatenate everything into one token stream per split.
6. Vocabulary: the 30,000 most frequent **training** tokens, plus `<pad>` and `<unk>` (30,002 entries). Everything else maps to `<unk>`, which covers 2.13% of training tokens.
7. Neural LMs read non-overlapping windows of 32 tokens, with targets shifted by one.

Resulting sizes: 2,124,061 train / 221,501 valid / 249,435 test tokens.

## How to run

Run the parts in order. Part II reuses the Part I vocabulary, and Part III uses LMs trained in Part I/II.

| Step | Notebook | Produces |
|---|---|---|
| Part I | `Part1_models_Colab.ipynb` (Colab: T4 runtime, then **Run all**) | see below |
| Part II | **TODO: notebook name** | **TODO** |
| Part III | **TODO: notebook name** | **TODO** |

`Part1_models_Colab.ipynb` mounts Google Drive and uses `/content/drive/MyDrive/dsa4213/ind_assignment` as its working directory. Change `PROJECT_DIR` in Cell 0 if your folder is different. When run locally, it uses the current directory.

### Part I outputs

| Report item | Cell | File |
|---|---|---|
| n-gram add-k tuning + test PPL | Cell 4 | — |
| Per-epoch training logs | Cells 7–9 | `results/history_{rnn,lstm,transformer}.json` |
| Perplexity / cost table | Cell 10 | `results/metrics_part1.csv` |
| Loss-curve figure | Cell 11 | `results/loss_curves_part1.png` |
| Generated-text comparison | Cell 12 | `results/generations.md` |

Checkpoints (`results/*.pt`) are saved but not committed.

### Expected runtime

| Step | T4 GPU (measured) | CPU |
|---|---|---|
| n-gram fit | 9 s | same (CPU-only) |
| RNN, 6 epochs (early stop) | 159 s | not measured |
| LSTM | 252 s | not measured |
| Transformer | 270 s | not measured |
| Part II | **TODO** | **TODO** |
| Part III | **TODO** | **TODO** |

On a T4, neural training in Part I takes about 11 minutes in total. The n-gram dev-set tuning (six CPU passes) adds a few minutes on top; it wasn't timed separately. Neural training is much slower on CPU.

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

### Generation (Cell 12)
- Temperature 0.8, top-k 40, at most 30 new tokens, context 32 tokens.
- Five fixed prompts (listed in the cell), the same decoding for every model.

### Part II — embedding ablation
- Settings: (a) random init, trainable; (b) own Skip-gram, frozen; (c) GloVe `glove-wiki-gigaword-100`, frozen.
- Embedding dim: 100, to match GloVe. **TODO:** Part I uses 128-d. If Part II's baseline is 100-d, say so, because the Part I model can't be reused as setting (a).
- Skip-gram (gensim `Word2Vec`, `sg=1`): vector_size 100, **TODO** window / min_count / negative / epochs / workers.
- OOV initialisation for words missing from GloVe: **TODO** (e.g. N(0, σ) with a fixed seed). Coverage is reported in the notebook.
- Other LM hyperparameters as in Part I (**TODO: confirm epochs** — Part I used max 15; update this if Part II uses 20).

### Part III — downstream classification
- Task: Rotten Tomatoes sentiment. Labels: **TODO**. Split: **TODO** ratios, stratified.
- Representations compared: **TODO** (e.g. last hidden state vs mean-pool).
- Classifier: scikit-learn `LogisticRegression`, C grid **TODO**, chosen on validation. Metrics: accuracy and macro-F1.

## Random seeds

| What | Seed |
|---|---|
| Part I neural LM training (init + shuffling) | 42 |
| Part II / III runs | 42, 43, 44 (mean ± std reported) |
| Generation / decoding | 123 |
| Rotten Tomatoes train/val/test split | **TODO** |

`set_seed()` seeds `random`, `numpy`, `torch` and CUDA. GPU kernels are not forced to be deterministic, so re-runs on a T4 can differ slightly in the last decimal places.

## Repository layout

```
Part1_models_Colab.ipynb   Part I (canonical; outputs kept so results are visible)
data/prepare_data.py       WikiText-2 download → data/processed/
src/evaluate.py            CE / PPL helper used by src/part1_models/RNN.py
src/part1_models/          early standalone scripts and a superseded local draft notebook;
                           the reported results come from Part1_models_Colab.ipynb
requirements.txt
```

## Attribution

- The n-gram LM, tokenizer and evaluation code are adapted from the DSA4213 Lecture 1 notebook. The RNN, LSTM and Transformer LMs are adapted from the DSA4213 Lecture 3 notebook. Each adapted file says so in its header.
- Datasets: WikiText-2 (Merity et al., 2016; CC BY-SA 3.0); Rotten Tomatoes (Kaggle, see above). Pretrained embeddings: GloVe 6B (Pennington et al., 2014), via gensim-data.
- AI coding assistance was used; see the report for details.
