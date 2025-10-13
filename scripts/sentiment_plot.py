"""
Анализ тональности
"""

import pandas as pd
import re
from transformers import pipeline
from tqdm import tqdm
import os


MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
INPUT_PATH = "data/messages.csv"
OUTPUT_PATH = "data/analysis_results.csv"
BATCH_SIZE = 16 


def clean_text(text: str) -> str:
    """Удаляет ссылки, хэштеги, упоминания, эмодзи и спецсимволы."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^а-яёa-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_sentiment():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Файл {INPUT_PATH} не найден!")

    print(f"Загружаем данные из {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["text"])
    df["clean_text"] = df["text"].apply(clean_text)

    print(f"Загружаем модель {MODEL_NAME}...")
    sentiment_model = pipeline("sentiment-analysis", model=MODEL_NAME)

    print("Анализ тональности...")
    tqdm.pandas()

    results = []
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df["clean_text"].iloc[i:i+BATCH_SIZE].astype(str).str.slice(0, 512).tolist()
        try:
            preds = sentiment_model(batch)
        except Exception as e:
            print(f"Ошибка при обработке батча {i}: {e}")
            preds = [{"label": "ERROR", "score": 0.0}] * len(batch)
        results.extend(preds)

    df["sentiment"] = [r["label"] for r in results]
    df["confidence"] = [r["score"] for r in results]

    sentiment_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1, "ERROR": None}
    df["sentiment_value"] = df["sentiment"].map(sentiment_map)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Анализ завершён! Результаты сохранены в {OUTPUT_PATH}.")
    print(df[["text", "sentiment", "confidence"]].head())


if __name__ == "__main__":
    analyze_sentiment()
