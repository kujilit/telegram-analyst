import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv
from telethon import TelegramClient

from fetch_messages import collect_channel_messages
from sentiment_plot import analyze_sentiment


load_dotenv()

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")

if not API_ID or not API_HASH:
    raise RuntimeError("API_ID или API_HASH не заданы в .env")


app = FastAPI(
    title="Telegram analyst API,
    description="Анализ Telegram-каналов",
    version="1.0.0",
)


@app.post("/analyze")
async def analyze_channel(
    channel: str = Query(..., description="Username или ссылка на канал"),
    date_from: Optional[datetime] = Query(
        None, description="Начало периода (UTC)"
    ),
    date_to: Optional[datetime] = Query(
        None, description="Конец периода (UTC)"
    ),
    limit: Optional[int] = Query(
        500, description="Максимум сообщений (для защиты API)"
    ),
):
    """
    Основная точка входа:
    - собирает сообщения
    - анализирует тональность
    - возвращает JSON
    """

    client = TelegramClient("session", API_ID, API_HASH)

    try:
        async with client:
            df_messages = await collect_channel_messages(
                client=client,
                channel=channel,
                date_from=date_from,
                date_to=date_to,
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if df_messages.empty:
        return {
            "channel": channel,
            "messages": [],
            "count": 0,
        }

    df_messages = df_messages.head(limit)

    df_sentiment = analyze_sentiment(df_messages)

    result = df_sentiment.to_dict(orient="records")

    return {
        "channel": channel,
        "count": len(result),
        "data": result,
    }
