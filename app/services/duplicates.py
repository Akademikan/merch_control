"""Поиск повторно загруженных фото по перцептивному хэшу (pHash)."""
from __future__ import annotations

import io
from pathlib import Path

import imagehash
from PIL import Image


def compute_phash(source: bytes | str | Path) -> str:
    """pHash 8x8 в виде 16-значной hex-строки. Устойчив к пересжатию и небольшому ресайзу."""
    fp = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    with Image.open(fp) as img:
        return str(imagehash.phash(img.convert("RGB")))


def hamming(a: str, b: str) -> int:
    """Расстояние Хэмминга между двумя hex-хэшами (число различающихся бит)."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def nearest(phash: str, known: list[str]) -> int | None:
    """Минимальное расстояние Хэмминга до известных хэшей (None, если сравнивать не с чем)."""
    if not known:
        return None
    return min(hamming(phash, k) for k in known)
