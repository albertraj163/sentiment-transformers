# sentiment_transformers.py
# Simple demo to classify sentiment using Hugging Face transformers pipeline.
from transformers import pipeline
import pandas as pd

DATA_PATH = "sample_texts.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    df['label'] = df['text'].apply(lambda t: classifier(t)[0]['label'])
    df['score'] = df['text'].apply(lambda t: classifier(t)[0]['score'])
    print(df)
    df.to_csv('sentiment_results.csv', index=False)

if __name__ == "__main__":
    main()
