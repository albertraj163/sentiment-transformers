# sentiment_transformers.py
# CLI demo — classify sentiment using Hugging Face transformers pipeline.
from sentiment_engine import analyze_dataframe

DATA_PATH = "sample_texts.csv"
OUTPUT_PATH = "sentiment_results.csv"


def main():
    import pandas as pd

    df = pd.read_csv(DATA_PATH)
    results = analyze_dataframe(df)
    print(results.to_string(index=False))
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
