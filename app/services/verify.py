"""Проверки метаданных фото относительно визита.

Каждая проверка возвращает CheckResult со статусом:
  ok      — проверка пройдена,
  fail    — нарушение,
  unknown — данных для проверки нет (например, EXIF стёрт мессенджером).
«unknown» не равно «fail»: решающий модуль учитывает его отдельным признаком.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .duplicates import nearest
from .exif import PhotoMeta
from .geo import haversine_m

EDITOR_KEYWORDS = (
    "photoshop", "gimp", "lightroom", "snapseed", "picsart", "canva", "pixlr",
    "facetune", "afterlight", "paint", "editor", "exiftool",
)


@dataclass
class VisitContext:
    store_lat: float
    store_lon: float
    radius_m: float
    tz: str  # часовой пояс магазина, например "Europe/Moscow"
    window_start: datetime  # окно визита из маршрутного листа (aware)
    window_end: datetime
    received_at: datetime  # когда фото получил сервер/Telegram (aware)


@dataclass
class CheckResult:
    name: str
    status: str  # ok | fail | unknown
    value: float | None = None
    detail: str = ""


def _parse_offset(offset: str) -> timezone | None:
    try:
        sign = -1 if offset.startswith("-") else 1
        hh, mm = offset.lstrip("+-").split(":")
        return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))
    except ValueError:
        return None


def capture_time_utc(meta: PhotoMeta, store_tz: str) -> tuple[datetime | None, str]:
    """Время съёмки в UTC и источник: gps | exif+offset | exif+store_tz | none.

    Приоритет у GPS-времени: оно в UTC и не зависит от часового пояса телефона.
    """
    if meta.gps_time_utc:
        return meta.gps_time_utc, "gps"
    if meta.taken_local:
        tz = _parse_offset(meta.utc_offset) if meta.utc_offset else None
        if tz is not None:
            return meta.taken_local.replace(tzinfo=tz).astimezone(timezone.utc), "exif+offset"
        local = meta.taken_local.replace(tzinfo=ZoneInfo(store_tz))
        return local.astimezone(timezone.utc), "exif+store_tz"
    return None, "none"


def check_location(meta: PhotoMeta, ctx: VisitContext, eps_gps_m: float = 100.0) -> CheckResult:
    """d <= r_магазина + eps_gps. eps подбирается по реальным данным (в помещении GPS шумит)."""
    if meta.gps_lat is None or meta.gps_lon is None:
        return CheckResult("location", "unknown", None, "в EXIF нет координат")
    d = haversine_m(meta.gps_lat, meta.gps_lon, ctx.store_lat, ctx.store_lon)
    limit = ctx.radius_m + eps_gps_m
    status = "ok" if d <= limit else "fail"
    return CheckResult("location", status, d, f"{d:.0f} м от магазина (допуск {limit:.0f} м)")


def check_in_window(meta: PhotoMeta, ctx: VisitContext, tolerance: timedelta = timedelta(minutes=10)) -> CheckResult:
    """Время съёмки лежит в окне визита [t_начала, t_конца] с допуском."""
    t, src = capture_time_utc(meta, ctx.tz)
    if t is None:
        return CheckResult("time_window", "unknown", None, "в EXIF нет времени съёмки")
    lo, hi = ctx.window_start - tolerance, ctx.window_end + tolerance
    if lo <= t <= hi:
        return CheckResult("time_window", "ok", 0.0, f"время съёмки в окне визита (источник: {src})")
    dev = (lo - t if t < lo else t - hi).total_seconds() / 60.0
    return CheckResult("time_window", "fail", dev, f"вне окна визита на {dev:.0f} мин (источник: {src})")


def check_upload_delay(
    meta: PhotoMeta,
    ctx: VisitContext,
    max_delay: timedelta = timedelta(hours=3),
    clock_skew: timedelta = timedelta(minutes=10),
) -> CheckResult:
    """Разница между временем получения сервером и временем съёмки."""
    t, src = capture_time_utc(meta, ctx.tz)
    if t is None:
        return CheckResult("upload_delay", "unknown", None, "в EXIF нет времени съёмки")
    delay = ctx.received_at - t
    minutes = delay.total_seconds() / 60.0
    if -clock_skew <= delay <= max_delay:
        return CheckResult("upload_delay", "ok", minutes, f"загружено через {minutes:.0f} мин после съёмки")
    if delay < -clock_skew:
        return CheckResult("upload_delay", "fail", minutes, "время съёмки позже времени получения (часы телефона?)")
    return CheckResult("upload_delay", "fail", minutes, f"загружено через {minutes:.0f} мин после съёмки")


def check_software(meta: PhotoMeta) -> CheckResult:
    """Следы графических редакторов в поле Software."""
    if not meta.software:
        return CheckResult("software", "unknown", None, "поле Software пусто")
    low = meta.software.lower()
    hit = next((k for k in EDITOR_KEYWORDS if k in low), None)
    if hit:
        return CheckResult("software", "fail", 1.0, f"Software: {meta.software}")
    return CheckResult("software", "ok", 0.0, f"Software: {meta.software}")


def check_series_consistency(metas: list[PhotoMeta]) -> CheckResult:
    """В серии фото одного визита должен быть один и тот же телефон (Make/Model)."""
    devices = {(m.make, m.model) for m in metas if m.make or m.model}
    if len(devices) <= 1:
        return CheckResult("device_consistency", "ok", float(len(devices)), "одно устройство в серии")
    return CheckResult("device_consistency", "fail", float(len(devices)), f"устройств в серии: {len(devices)}")


def check_speed(metas_with_time: list[tuple[datetime, float, float]], max_kmh: float = 150.0) -> CheckResult:
    """Скорость перемещения между последовательными точками съёмки.

    Ловит «перелёт» между городами. Внутри магазина GPS шумит, поэтому порог высокий.
    Вход: список (время UTC, широта, долгота).
    """
    pts = sorted(metas_with_time)
    worst = 0.0
    for (t1, la1, lo1), (t2, la2, lo2) in zip(pts, pts[1:]):
        dt = (t2 - t1).total_seconds()
        if dt < 1:
            continue
        worst = max(worst, haversine_m(la1, lo1, la2, lo2) / dt * 3.6)
    if len(pts) < 2:
        return CheckResult("speed", "unknown", None, "меньше двух фото с координатами и временем")
    status = "ok" if worst <= max_kmh else "fail"
    return CheckResult("speed", status, worst, f"макс. скорость между снимками {worst:.0f} км/ч")


def check_duplicate(phash: str, known_hashes: list[str], threshold: int = 5) -> CheckResult:
    """Фото почти совпадает с ранее загруженным (расстояние Хэмминга <= threshold)."""
    d = nearest(phash, known_hashes)
    if d is None:
        return CheckResult("duplicate", "ok", None, "сравнивать не с чем")
    if d <= threshold:
        return CheckResult("duplicate", "fail", float(d), f"похоже на ранее загруженное (Хэмминг {d})")
    return CheckResult("duplicate", "ok", float(d), f"ближайшее расстояние Хэмминга {d}")


def run_all_checks(
    meta: PhotoMeta,
    ctx: VisitContext,
    phash: str,
    known_hashes: list[str],
    series: list[PhotoMeta],
    series_points: list[tuple[datetime, float, float]],
    eps_gps_m: float = 100.0,
    dup_threshold: int = 5,
) -> dict[str, CheckResult]:
    results = [
        check_location(meta, ctx, eps_gps_m),
        check_in_window(meta, ctx),
        check_upload_delay(meta, ctx),
        check_software(meta),
        check_series_consistency(series + [meta]),
        check_speed(series_points),
        check_duplicate(phash, known_hashes, dup_threshold),
    ]
    return {r.name: r for r in results}
