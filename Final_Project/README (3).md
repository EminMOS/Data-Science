# Twitter Sentiment Analysis Project

This folder contains a complete sentiment-analysis pipeline for the Kaggle/Mendeley Twitter Sentiments dataset. The project compares classical ML models with a CNN-LSTM hybrid, supports configurable sample sizes for faster experimentation, and produces ready-to-use visualizations plus serialized models.

## Dataset

- **File**: `Twitter_Data.csv`
- **Rows**: ~162,980 labeled tweets with integer sentiment targets (-1, 0, 1)
- **Source**: Hussein, Sherif (2021). “Twitter Sentiments Dataset”, Mendeley, doi:10.17632/z9zw7nt5h2.1

The CSV ships pre-cleaned (no raw download step). Use `--data_path` to point to a different copy if needed.

## Repository Contents

| Item | Description |
| --- | --- |
| `twitter_sentiment_main.py` | Primary end-to-end pipeline (EDA → features → models → evaluation) |
| `README (3).md` | This guide |
| `requirements.txt` | Minimal dependency pin set for Python 3.8+ |
| `Twitter_Data.csv` | Main dataset |
| `benchmark_sample_sizes.py` (optional) | Helper that measures runtime/accuracy over multiple sample sizes via subprocess calls to the main script |
| `test_100_samples.py` (optional) | Convenience runner that demonstrates the 100-sample quick-test mode |

Helper scripts are not required for day-to-day runs but can be useful references.

## Environment Setup

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow is the heaviest dependency; expect a few minutes to install and ensure that the runtime has enough RAM/GPU if you switch to full-dataset training.

## Running the Pipeline

The script is fully CLI-driven:

```bash
python twitter_sentiment_main.py [OPTIONS]
```

| Option | Purpose |
| --- | --- |
| `--data_path PATH` | Custom CSV location (defaults to `Twitter_Data.csv` adjacent to the script) |
| `--sample-size N` | Number of rows to use. Set `N <= 0` for the full dataset. Omit to use the default of **100 samples** for fast experiments. |
| `--dev` | Development mode (forces simplified hyperparameters and a 1,000-row sample unless `--sample-size` overrides it). |
| `--random_state N` | Seed for reproducible splits (defaults to 42). |

### Quick Start

```bash
# Ultra-fast smoke test (~30 s)
python twitter_sentiment_main.py                   # default 100-row sample

# Explicit custom sample size
python twitter_sentiment_main.py --sample-size 500

# Development profile (1,000 rows + lightweight hyperparameters)
python twitter_sentiment_main.py --dev

# Full production-style training (~15–30 min on CPU)
python twitter_sentiment_main.py --sample-size 0
```

Sampling uses stratified selection when class counts allow it and falls back to random sampling otherwise. The script prints the chosen mode and dataset distribution at run time so you always know what subset is in play.

## Pipeline Overview

1. **EDA** – Sentiment distribution, tweet length stats, and visualization generation.
2. **Preprocessing** – Cleans mentions, URLs, hashtags, punctuation, lowercases, and removes a curated stopword list (no NLTK downloads required).
3. **Feature Engineering**
   - TF-IDF vectors (max features scale with dataset size).
   - Word2Vec embeddings averaged per tweet.
   - Tokenized + padded sequences for neural models.
4. **Classical Models** – Decision Tree, KNN, Logistic Regression, and a soft Voting ensemble tuned with GridSearchCV. Hyperparameter grids shrink automatically for small samples/development runs.
5. **Neural Model** – CNN + LSTM hybrid with adjustable vocabulary, embedding size, epochs, and batch size dictated by dataset size; includes EarlyStopping and ReduceLROnPlateau callbacks.
6. **Evaluation** – Accuracy, precision, recall, macro F1, confusion matrices, ROC curves, and a consolidated comparison table highlighting the best-performing model.

## Sample Size Profiles

| Samples | Typical Runtime (CPU) | Recommended Use |
| --- | --- | --- |
| 100 (default) | ~30 seconds | Smoke tests, debugging |
| 500 | ~1–2 minutes | Feature tweaks |
| 1,000 (`--dev`) | ~2–3 minutes | Balanced dev loop |
| 2,500 | ~4–5 minutes | More stable accuracy estimates |
| 5,000 | ~6–8 minutes | Near-final validation |
| Full (162K) | 15–30 minutes | Final training/reporting |

## Outputs

The script writes artifacts into the working directory:

- Visuals: `eda_visualization.png`, `model_comparison.png`, `roc_curves.png`, `nn_training_history.png`, `confusion_matrix_<model>.png`
- Tabular summaries: `model_comparison.csv`
- Saved assets (within `models/` if configured inside the script): TF-IDF vectorizer, Word2Vec model, neural tokenizer, best NN weights (`best_model.h5`), and pickled classical estimators.

## Helper Scripts

- `benchmark_sample_sizes.py` – Runs the main script sequentially across predefined sample sizes, captures execution time/accuracy, and can plot a runtime curve (`benchmark_results.png`). Useful when tuning default sample sizes or sharing performance notes.
- `test_100_samples.py` – Executes a single `--sample-size 100` run, highlights key log sections, and reports elapsed time. Handy for demos or CI smoke checks.

These wrappers simply call `twitter_sentiment_main.py` via `subprocess.run`; delete or modify them freely if you don’t need the automation.

## Troubleshooting & Tips

- **Memory pressure**: Lower TF-IDF `max_features` or reduce `batch_size` in the neural model when operating on modest hardware.
- **Runtime too long**: Drop the sample size, use `--dev`, or comment out the neural model section when iterating on classical models only.
- **Reproducibility**: Always set `--random_state` for consistent stratified samples and weight initialization.
- **Custom datasets**: Point `--data_path` to your file and ensure it exposes the same `category` and `clean_text` columns (or update the script accordingly).

## Citation

Hussein, Sherif (2021). *Twitter Sentiments Dataset*. Mendeley Data, V1. https://doi.org/10.17632/z9zw7nt5h2.1
