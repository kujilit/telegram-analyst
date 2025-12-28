import re
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
BATCH_SIZE = 16

POSITIVE_EMOJI = "😂😄😅😊😍😁🙂"
NEGATIVE_EMOJI = "😡😠😢😭🤮😤😞"


def extract_text_features(text: str) -> dict:
    text = str(text)

    exclamations = text.count("!")
    questions = text.count("?")

    caps_letters = sum(1 for c in text if c.isupper())
    letters = sum(1 for c in text if c.isalpha())
    caps_ratio = caps_letters / letters if letters else 0

    repeated_letters = len(re.findall(r"(.)\1{2,}", text))

    emoji_positive = sum(c in POSITIVE_EMOJI for c in text)
    emoji_negative = sum(c in NEGATIVE_EMOJI for c in text)

    return {
        "exclamations": exclamations,
        "questions": questions,
        "caps_ratio": caps_ratio,
        "repeated_letters": repeated_letters,
        "emoji_positive": emoji_positive,
        "emoji_negative": emoji_negative,
    }


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^а-яёa-z!? ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def heuristic_sentiment_score(features: dict) -> float:
    score = 0.0

    score += features["emoji_positive"] * 0.3
    score -= features["emoji_negative"] * 0.4

    score += min(features["exclamations"], 5) * 0.05
    score += features["caps_ratio"] * 0.5
    score += features["repeated_letters"] * 0.1

    return max(min(score, 1.0), -1.0)


def analyze_sentiment(
    df: pd.DataFrame,
    *,
    text_column: str = "text",
    batch_size: int = BATCH_SIZE,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:

    if text_column not in df.columns:
        raise ValueError(f"В DataFrame нет колонки '{text_column}'")

    work_df = df.copy()
    work_df = work_df.dropna(subset=[text_column])

    # Извлечение фич
    features = work_df[text_column].apply(extract_text_features)
    features_df = pd.DataFrame(features.tolist())
    work_df = pd.concat([work_df.reset_index(drop=True), features_df], axis=1)

    work_df["clean_text"] = work_df[text_column].apply(clean_text)

    sentiment_model = pipeline(
        "sentiment-analysis",
        model=model_name
    )

    texts = (
        work_df["clean_text"]
        .astype(str)
        .str.slice(0, 512)
        .tolist()
    )

    results = []

    tqdm.pandas(desc="Sentiment analysis")

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        try:
            preds = sentiment_model(batch)
        except Exception:
            preds = [{"label": "ERROR", "score": 0.0}] * len(batch)
        results.extend(preds)

    work_df["sentiment"] = [r["label"] for r in results]
    work_df["confidence"] = [r["score"] for r in results]

    sentiment_map = {
        "POSITIVE": 1,
        "NEUTRAL": 0,
        "NEGATIVE": -1,
        "ERROR": 0,
    }

    work_df["model_score"] = (
        work_df["sentiment"].map(sentiment_map)
        * work_df["confidence"]
    )

    work_df["heuristic_score"] = work_df.apply(
        lambda row: heuristic_sentiment_score(row.to_dict()),
        axis=1
    )

    work_df["final_sentiment_score"] = (
        work_df["model_score"] * 0.7
        + work_df["heuristic_score"] * 0.3
    )

    return work_df
