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

## Quick start

```bash
# Clone the repository
git clone https://github.com/albertraj163/sentiment-transformers.git
cd sentiment-transformers

# Install dependencies
pip install -r requirements.txt

# Launch the web UI
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

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
