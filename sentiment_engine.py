"""Shared sentiment analysis engine using Hugging Face DistilBERT."""

from functools import lru_cache
from typing import Any

import pandas as pd
from transformers import pipeline

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


@lru_cache(maxsize=1)
def get_classifier(model_name: str = DEFAULT_MODEL):
    """Load and cache the sentiment classifier pipeline."""
    return pipeline("sentiment-analysis", model=model_name)


def analyze_text(text: str, model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Classify sentiment for a single text string."""
    text = (text or "").strip()
    if not text:
        return {"label": "NEUTRAL", "score": 0.0, "text": text}

    result = get_classifier(model_name)(text)[0]
    return {
        "text": text,
        "label": result["label"],
        "score": float(result["score"]),
    }


def analyze_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    model_name: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Classify sentiment for every row in a DataFrame."""
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in input data.")

    classifier = get_classifier(model_name)
    results = df[text_column].astype(str).apply(lambda t: classifier(t.strip())[0])

    output = df.copy()
    output["label"] = results.apply(lambda r: r["label"])
    output["score"] = results.apply(lambda r: float(r["score"]))
    output["confidence_pct"] = (output["score"] * 100).round(1)
    return output
