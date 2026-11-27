"""Генерация тестовых JPEG с заданным EXIF (для тестов и demo-скрипта)."""
from __future__ import annotations

import io
import random
from datetime import datetime

import piexif
from PIL import Image, ImageDraw


def _rat(x: float, scale: int = 1_000_000) -> tuple[int, int]:
    return int(round(x * scale)), scale


def _dms(deg: float):
    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return (d, 1), (m, 1), _rat(s)


def make_jpeg(
    lat: float | None = None,
    lon: float | None = None,
    taken_local: datetime | None = None,
    gps_utc: datetime | None = None,
    make: str | None = "Samsung",
    model: str | None = "SM-A546B",
    software: str | None = None,
    offset: str | None = None,
    seed: int = 0,
    with_exif: bool = True,
) -> bytes:
    """seed влияет на содержимое картинки (разные seed -> разные pHash)."""
    rnd = random.Random(seed)
    img = Image.new("RGB", (320, 240), (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)))
    draw = ImageDraw.Draw(img)
    for _ in range(12):
        x, y = rnd.randint(0, 280), rnd.randint(0, 200)
        draw.rectangle([x, y, x + rnd.randint(10, 80), y + rnd.randint(10, 60)],
                       fill=(rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)))
    buf = io.BytesIO()
    if not with_exif:
        img.save(buf, "JPEG")
        return buf.getvalue()

    zeroth, exif, gps = {}, {}, {}
    if make:
        zeroth[piexif.ImageIFD.Make] = make.encode()
    if model:
        zeroth[piexif.ImageIFD.Model] = model.encode()
    if software:
        zeroth[piexif.ImageIFD.Software] = software.encode()
    if taken_local:
        exif[piexif.ExifIFD.DateTimeOriginal] = taken_local.strftime("%Y:%m:%d %H:%M:%S").encode()
    if offset:
        exif[piexif.ExifIFD.OffsetTimeOriginal] = offset.encode()
    if lat is not None and lon is not None:
        gps[piexif.GPSIFD.GPSLatitudeRef] = b"N" if lat >= 0 else b"S"
        gps[piexif.GPSIFD.GPSLatitude] = _dms(lat)
        gps[piexif.GPSIFD.GPSLongitudeRef] = b"E" if lon >= 0 else b"W"
        gps[piexif.GPSIFD.GPSLongitude] = _dms(lon)
    if gps_utc:
        gps[piexif.GPSIFD.GPSDateStamp] = gps_utc.strftime("%Y:%m:%d").encode()
        gps[piexif.GPSIFD.GPSTimeStamp] = ((gps_utc.hour, 1), (gps_utc.minute, 1), (gps_utc.second, 1))
    img.save(buf, "JPEG", exif=piexif.dump({"0th": zeroth, "Exif": exif, "GPS": gps}))
    return buf.getvalue()
