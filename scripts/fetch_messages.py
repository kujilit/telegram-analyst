import os
import pandas as pd
from datetime import datetime, timezone
from telethon import TelegramClient
from dotenv import load_dotenv
import asyncio

load_dotenv()

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")

CLIENT = TelegramClient("session", API_ID, API_HASH)

DATE_FROM = datetime(2025, 1, 1, tzinfo=timezone.utc)
DATE_TO = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


async def collect_channel_messages(
    client: TelegramClient,
    channel: str,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> pd.DataFrame:

    rows = []

    async for message in client.iter_messages(channel, offset_date=date_to):
        if not message.date:
            continue

        if date_from and message.date < date_from:
            break

        text = message.text or ""

        rows.append({
            "id": message.id,
            "date": message.date,
            "text": text,
            "views": message.views,
            "forwards": message.forwards,
            "replies": message.replies.replies if message.replies else 0,
            "reactions": sum(r.count for r in message.reactions.results)
                if message.reactions else 0,
            "has_media": message.media is not None,
            "text_length": len(text),
            "word_count": len(text.split()),
            "hour": message.date.hour,
            "weekday": message.date.weekday(),
        })

    return pd.DataFrame(rows)


async def main(cahnnel_name: str):
    client = TelegramClient("session", API_ID, API_HASH)

    async with client:
        return await collect_channel_messages(
            client=client,
            channel=cahnnel_name,
            date_from=DATE_FROM,
            date_to=DATE_TO,
        )


if __name__ == "__main__":
    channel = "your_channel_name"
    df = asyncio.run(main(channel))
    print(df)