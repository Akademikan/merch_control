"""Извлечение метаданных (EXIF) из фотографии.

Используется только Pillow, чтобы не тянуть лишних зависимостей.
Важно: EXIF легко подделать или стереть, поэтому результат этого модуля —
это «признаки» для решающего модуля, а не доказательство подлинности.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

try:  # HEIC с iPhone (необязательно): pip install pillow-heif
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

EXIF_IFD = 0x8769  # указатель на подблок Exif
GPS_IFD = 0x8825  # указатель на блок GPS

# Теги основного блока
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_SOFTWARE = 0x0131
TAG_DATETIME = 0x0132
# Теги блока Exif
TAG_DATETIME_ORIGINAL = 0x9003
TAG_OFFSET_TIME_ORIGINAL = 0x9011
# Теги блока GPS
GPS_LAT_REF, GPS_LAT, GPS_LON_REF, GPS_LON = 1, 2, 3, 4
GPS_TIME, GPS_DATE = 7, 29


@dataclass
class PhotoMeta:
    has_exif: bool = False
    taken_local: datetime | None = None  # время съёмки как записано в EXIF (без пояса)
    utc_offset: str | None = None  # например "+03:00", если телефон его записал
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_time_utc: datetime | None = None  # время по спутникам, всегда UTC
    make: str | None = None
    model: str | None = None
    software: str | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("taken_local", "gps_time_utc"):
            if d[k] is not None:
                d[k] = d[k].isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhotoMeta":
        d = dict(d)
        for k in ("taken_local", "gps_time_utc"):
            if d.get(k):
                d[k] = datetime.fromisoformat(d[k])
        return cls(**d)


def _text(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="ignore")
    v = str(v).replace("\x00", "").strip()
    return v or None


def _dms_to_deg(dms: Any, ref: Any) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if _text(ref) in ("S", "W"):
        deg = -deg
    return deg


def _parse_exif_dt(value: Any) -> datetime | None:
    value = _text(value)
    if not value:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _parse_gps_dt(date: Any, ts: Any) -> datetime | None:
    date = _text(date)
    if not date or not ts:
        return None
    try:
        h, m, s = (float(x) for x in ts)
        y, mo, d = (int(x) for x in date.split(":"))
        return datetime(y, mo, d, int(h), int(m), int(s), tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def read_metadata(source: bytes | str | Path) -> PhotoMeta:
    """Читает EXIF из байтов или файла. Не падает на фото без EXIF."""
    fp = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    with Image.open(fp) as img:
        meta = PhotoMeta(width=img.width, height=img.height)
        exif = img.getexif()
        if not exif:
            return meta
        exif_ifd = exif.get_ifd(EXIF_IFD)
        gps_ifd = exif.get_ifd(GPS_IFD)

    meta.has_exif = True
    meta.make = _text(exif.get(TAG_MAKE))
    meta.model = _text(exif.get(TAG_MODEL))
    meta.software = _text(exif.get(TAG_SOFTWARE))
    meta.taken_local = _parse_exif_dt(exif_ifd.get(TAG_DATETIME_ORIGINAL)) or _parse_exif_dt(
        exif.get(TAG_DATETIME)
    )
    meta.utc_offset = _text(exif_ifd.get(TAG_OFFSET_TIME_ORIGINAL))

    if gps_ifd:
        meta.gps_lat = _dms_to_deg(gps_ifd.get(GPS_LAT), gps_ifd.get(GPS_LAT_REF))
        meta.gps_lon = _dms_to_deg(gps_ifd.get(GPS_LON), gps_ifd.get(GPS_LON_REF))
        meta.gps_time_utc = _parse_gps_dt(gps_ifd.get(GPS_DATE), gps_ifd.get(GPS_TIME))
        # координаты (0, 0) — типичный «пустой» GPS
        if meta.gps_lat == 0.0 and meta.gps_lon == 0.0:
            meta.gps_lat = meta.gps_lon = None
    return meta
