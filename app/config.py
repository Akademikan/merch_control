from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/merch.db"
    upload_dir: str = "./data/uploads"
    # Допуск по GPS сверх радиуса магазина, м. Подбирается экспериментально.
    eps_gps_m: float = 100.0
    # Порог расстояния Хэмминга для дубликатов
    dup_threshold: int = 5
    # Границы решающего модуля
    p_low: float = 0.25
    p_high: float = 0.75
    # Файл с обученными весами (необязательно) и веса детектора (необязательно)
    weights_path: str | None = "./data/decision_weights.json"
    detector_weights: str | None = "./data/detector.pt"

    # Telegram-бот
    bot_token: str = ""
    api_url: str = "http://127.0.0.1:8000"
