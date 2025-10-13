"""
Сбор сообщений
"""

import os
import pandas as pd
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
CHANNEL = os.getenv("CHANNEL")
OUTPUT_PATH = "data/messages.csv"

if not API_ID or not API_HASH:
    raise ValueError("❌ Не заданы API_ID или API_HASH в файле .env")


client = TelegramClient('session', API_ID, API_HASH)

async def fetch_messages():
    print(f"Подключаемся к каналу: {CHANNEL}")
    messages_data = []

    async for message in client.iter_messages(CHANNEL):
        if message.text:
            messages_data.append({
                "id": message.id,
                "date": message.date,
                "text": message.text
            })

    df = pd.DataFrame(messages_data)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранено {len(df)} сообщений в {OUTPUT_PATH}")


if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(fetch_messages())
