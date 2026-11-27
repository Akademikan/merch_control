"""Решающий модуль: логистическая регрессия + две границы.

p = 1 / (1 + exp(-(w0 + w·x)))   — вероятность того, что отчёт нарушен/подозрителен.
  p < p_low   -> accept (принять автоматически)
  p > p_high  -> reject (отклонить)
  иначе       -> manual (на ручную проверку супервайзеру)

ВАЖНО: веса по умолчанию — стартовые экспертные значения, чтобы система работала
«из коробки». Для диплома их нужно обучить на размеченных отчётах:
см. scripts/train_decision.py. Пороги p_low/p_high подбираются по ROC и стоимости ошибок.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .verify import CheckResult

FEATURES = (
    "loc_fail",           # фото вне допустимого радиуса магазина
    "loc_far_km",         # насколько далеко (км, не более 10)
    "time_window_fail",   # время съёмки вне окна визита
    "upload_delay_fail",  # слишком большая задержка загрузки / часы телефона
    "no_exif",            # EXIF отсутствует вовсе
    "no_gps",             # нет координат
    "duplicate",          # похоже на ранее загруженное фото
    "software_edit",      # следы графического редактора
    "multi_device",       # в серии несколько устройств
    "speed_fail",         # нереальная скорость перемещения
    "missing_share",      # доля отсутствующих SKU по чек-листу (1 - Compl)
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "bias": -2.5,
    "loc_fail": 3.0,
    "loc_far_km": 0.3,
    "time_window_fail": 3.0,
    "upload_delay_fail": 1.5,
    "no_exif": 1.2,
    "no_gps": 0.8,
    "duplicate": 4.0,
    "software_edit": 2.0,
    "multi_device": 1.5,
    "speed_fail": 3.0,
    "missing_share": 2.5,
}

REASON_TEXT = {
    "loc_fail": "фото сделано не в магазине",
    "time_window_fail": "время съёмки не совпадает с визитом",
    "upload_delay_fail": "подозрительная задержка между съёмкой и загрузкой",
    "no_exif": "в файле нет метаданных (вероятно, фото отправлено как сжатое изображение)",
    "no_gps": "в метаданных нет координат",
    "duplicate": "фото похоже на ранее загруженное",
    "software_edit": "в метаданных есть следы графического редактора",
    "multi_device": "в серии фото разные устройства",
    "speed_fail": "нереальная скорость перемещения между снимками",
    "missing_share": "на фото найдены не все товары из чек-листа",
}


def build_features(
    checks: dict[str, CheckResult],
    has_exif: bool,
    completeness: float | None = None,
) -> dict[str, float]:
    def failed(name: str) -> float:
        return 1.0 if name in checks and checks[name].status == "fail" else 0.0

    loc = checks.get("location")
    loc_far = 0.0
    if loc and loc.status == "fail" and loc.value is not None:
        loc_far = min(loc.value / 1000.0, 10.0)
    return {
        "loc_fail": failed("location"),
        "loc_far_km": loc_far,
        "time_window_fail": failed("time_window"),
        "upload_delay_fail": failed("upload_delay"),
        "no_exif": 0.0 if has_exif else 1.0,
        "no_gps": 1.0 if (loc is None or loc.status == "unknown") else 0.0,
        "duplicate": failed("duplicate"),
        "software_edit": failed("software"),
        "multi_device": failed("device_consistency"),
        "speed_fail": failed("speed"),
        "missing_share": 0.0 if completeness is None else max(0.0, min(1.0, 1.0 - completeness)),
    }


def load_weights(path: str | Path | None) -> dict[str, float]:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(DEFAULT_WEIGHTS)


def probability(features: dict[str, float], weights: dict[str, float] | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    z = w.get("bias", 0.0) + sum(w.get(k, 0.0) * v for k, v in features.items())
    return 1.0 / (1.0 + math.exp(-z))


def decide(p: float, p_low: float = 0.25, p_high: float = 0.75) -> str:
    if p < p_low:
        return "accept"
    if p > p_high:
        return "reject"
    return "manual"


def reasons(features: dict[str, float], weights: dict[str, float] | None = None) -> list[str]:
    """Причины решения: признаки с положительным вкладом, по убыванию вклада."""
    w = weights or DEFAULT_WEIGHTS
    contrib = [(w.get(k, 0.0) * v, k) for k, v in features.items() if v > 0 and k in REASON_TEXT]
    contrib.sort(reverse=True)
    return [REASON_TEXT[k] for c, k in contrib if c > 0]
