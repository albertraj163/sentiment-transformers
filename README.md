# Sentiment Analysis using Transformers

Professional AI sentiment analysis demo using **Hugging Face DistilBERT** with a Streamlit web UI.

| Resource | Link |
|----------|------|
| **GitHub Repository** | https://github.com/albertraj163/sentiment-transformers |
| **Hugging Face Model** | https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english |
| **Local App URL** | http://localhost:8501 |

## Features

- **Web UI** — Clean Streamlit dashboard for single text, batch CSV, and sample analysis
- **DistilBERT SST-2** — Fast POSITIVE / NEGATIVE classification with confidence scores
- **Batch processing** — Upload CSV, view metrics, download results
- **CLI script** — Command-line workflow for automation

## Quick start

```bash
git clone https://github.com/albertraj163/sentiment-transformers.git
cd sentiment-transformers
pip install -r requirements.txt
./run.sh
```

Then open the URL printed in your terminal (usually **http://localhost:8501**).

Alternative:

```bash
streamlit run app.py
```

## Project structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `sentiment_engine.py` | Shared sentiment analysis engine |
| `sentiment_transformers.py` | CLI batch script |
| `sample_texts.csv` | Example dataset |
| `run.sh` | One-command local launcher |
| `requirements.txt` | Python dependencies |

## CLI usage

```bash
python sentiment_transformers.py
```

Reads `sample_texts.csv` and writes `sentiment_results.csv`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GitHub repo shows no app UI | Run locally with `./run.sh` — GitHub hosts code, not the live UI |
| Port already in use | `run.sh` auto-picks the next free port |
| `ModuleNotFoundError: pipeline` | Run `pip install -r requirements.txt` |
| Terminal torchvision warnings | Harmless; disabled via Streamlit file watcher config |

## Tech stack

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [DistilBERT SST-2 model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [Streamlit](https://streamlit.io)
- [PyTorch](https://pytorch.org)

## Author

**Albert Raj** — https://github.com/albertraj163

## License

MIT — free for personal and commercial use.
