# Sentiment Analysis using Transformers

Professional AI sentiment analysis demo using **Hugging Face DistilBERT** with a Streamlit web UI.

## Official links

| Resource | URL |
|----------|-----|
| GitHub Repository | https://github.com/albertraj163/sentiment-transformers |
| Hugging Face Model | https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english |
| **Local App (default port)** | **http://localhost:8501** |

> GitHub hosts the source code only. The web UI runs on your machine after `./run.sh`.

## Run locally (verified)

```bash
git clone https://github.com/albertraj163/sentiment-transformers.git
cd sentiment-transformers
pip install -r requirements.txt
./run.sh
```

After startup, open:

```
http://localhost:8501
```

If port `8501` is busy, `run.sh` uses the next free port (for example `8502`) and prints the exact link in the terminal.

Alternative:

```bash
streamlit run app.py --server.port 8501
```

## What you should see

1. Terminal prints `Local app : http://localhost:8501`
2. Browser opens the Sentiment Transformers dashboard
3. Tabs: **Single Text**, **Batch Upload**, **Sample Dataset**

## Features

- **Web UI** — Streamlit dashboard for single text, batch CSV, and sample analysis
- **DistilBERT SST-2** — POSITIVE / NEGATIVE classification with confidence scores
- **Batch processing** — Upload CSV, view metrics, download results
- **CLI script** — Command-line workflow for automation

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
| GitHub repo shows no app UI | Run `./run.sh` and open **http://localhost:8501** |
| Port already in use | Use the URL printed by `run.sh` (may be `8502`, `8503`, …) |
| `ModuleNotFoundError: pipeline` | Run `pip install -r requirements.txt` |
| Blank page | Confirm terminal shows `Uvicorn server started on 0.0.0.0:8501` |

## Tech stack

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [DistilBERT SST-2 model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [Streamlit](https://streamlit.io)
- [PyTorch](https://pytorch.org)

## Author

**Albert Raj** — https://github.com/albertraj163

## License

MIT — free for personal and commercial use.
