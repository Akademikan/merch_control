"""Telegram-бот приёма фотоотчётов (aiogram 3).

Запуск:  python -m app.bot.telegram_bot
Переменные окружения (.env): BOT_TOKEN, API_URL.

Логика:
  /visit <id>   — выбрать текущий визит
  фото файлом   — отправляется в API; EXIF сохраняется
  фото «сжатым» — Telegram стирает EXIF; бот просит переслать файлом (но всё равно обрабатывает)
Время получения берётся из message.date — это время сервера Telegram, ему можно доверять
больше, чем часам телефона.
"""
from __future__ import annotations

import asyncio
import io

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from ..config import Settings

cfg = Settings()
dp = Dispatcher()
current_visit: dict[int, int] = {}  # telegram user id -> visit id (для MVP в памяти; потом в БД)

VERDICT = {"accept": "принято", "reject": "отклонено", "manual": "отправлено на ручную проверку"}


@dp.message(Command("start"))
async def start(m: Message):
    await m.answer(
        "Выберите визит командой /visit <номер>, затем отправляйте фото выкладки "
        "ФАЙЛОМ (скрепка → Файл), иначе Telegram удалит метаданные съёмки."
    )


@dp.message(Command("visit"))
async def set_visit(m: Message, command: CommandObject):
    if not command.args or not command.args.strip().isdigit():
        await m.answer("Использование: /visit 12")
        return
    current_visit[m.from_user.id] = int(command.args.strip())
    await m.answer(f"Текущий визит: {command.args.strip()}")


async def forward_to_api(m: Message, file_id: str, warn_compressed: bool):
    visit_id = current_visit.get(m.from_user.id)
    if visit_id is None:
        await m.answer("Сначала выберите визит: /visit <номер>")
        return
    buf = io.BytesIO()
    await m.bot.download(file_id, destination=buf)
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{cfg.api_url}/visits/{visit_id}/photos",
            files={"file": ("photo.jpg", buf.getvalue())},
            data={"received_at": m.date.isoformat()},
        )
    if r.status_code != 200:
        await m.answer(f"Ошибка обработки: {r.text[:200]}")
        return
    res = r.json()
    text = f"Фото {VERDICT.get(res['decision'], res['decision'])} (p={res['probability']})."
    if res["reasons"]:
        text += "\nПричины:\n- " + "\n- ".join(res["reasons"])
    if warn_compressed:
        text += "\nФото отправлено сжатым, метаданные потеряны. Отправляйте файлом."
    await m.answer(text)


@dp.message(F.document & F.document.mime_type.startswith("image/"))
async def on_document(m: Message):
    await forward_to_api(m, m.document.file_id, warn_compressed=False)


@dp.message(F.photo)
async def on_photo(m: Message):
    await forward_to_api(m, m.photo[-1].file_id, warn_compressed=True)


async def main():
    if not cfg.bot_token:
        raise SystemExit("Задайте BOT_TOKEN в .env")
    await dp.start_polling(Bot(cfg.bot_token))


if __name__ == "__main__":
    asyncio.run(main())
