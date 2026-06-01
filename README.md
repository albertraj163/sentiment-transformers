# Sentiment Analysis using Transformers

AI-powered sentiment analysis demo using **Hugging Face Transformers** (DistilBERT) with a professional web UI.

**Repository:** [github.com/albertraj163/sentiment-transformers](https://github.com/albertraj163/sentiment-transformers)

## Features

- **Web UI** — Modern Streamlit interface with single-text, batch CSV, and sample analysis
- **DistilBERT** — Fast, accurate sentiment classification (POSITIVE / NEGATIVE)
- **Batch processing** — Upload CSV files and download results
- **CLI script** — Original command-line workflow still supported

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `sentiment_engine.py` | Shared sentiment analysis engine |
| `sentiment_transformers.py` | CLI script for batch analysis |
| `sample_texts.csv` | Example texts for testing |
| `requirements.txt` | Python dependencies |

## Quick start (local)

> **Note:** The GitHub repo link is for source code only. Opening the repo URL in a browser will show a **404** for the app itself. You must run the app locally (or deploy to Streamlit Cloud).

```bash
# Clone the repository
git clone https://github.com/albertraj163/sentiment-transformers.git
cd sentiment-transformers

# Install dependencies
pip install -r requirements.txt

# Launch the web UI (choose one)
./run.sh
# or
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Troubleshooting 404

| Problem | Fix |
|---------|-----|
| Opened GitHub repo URL | Run locally with `./run.sh` — app is not hosted on GitHub |
| Page not loading | Make sure terminal shows `Local URL: http://localhost:8501` |
| Wrong folder | Run from repo root where `app.py` exists |
| Port busy | Use `streamlit run app.py --server.port 8502` |
| `ModuleNotFoundError: pipeline` | Run `pip install -r requirements.txt` (old Pillow breaks transformers) |

## CLI usage

```bash
python sentiment_transformers.py
```

Reads `sample_texts.csv` and writes results to `sentiment_results.csv`.

## Tech stack

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — `distilbert-base-uncased-finetuned-sst-2-english`
- [Streamlit](https://streamlit.io) — Web UI (free & open source)
- [PyTorch](https://pytorch.org) — Deep learning backend
- [Pandas](https://pandas.pydata.org) — Data handling

## License

MIT — free for personal and commercial use.
