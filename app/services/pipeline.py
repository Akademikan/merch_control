"""Конвейер обработки одного фото: сохранить -> метаданные -> хэш -> проверки -> детектор -> решение."""
from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image
import io
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import Photo, Visit, aware
from . import decision as dec
from .detector import Detector, completeness
from .duplicates import compute_phash
from .exif import PhotoMeta, read_metadata
from .verify import VisitContext, capture_time_utc, run_all_checks


def save_original(content: bytes, upload_dir: str | Path, visit_id: int) -> Path:
    """Оригинал сохраняется без перекодирования, иначе EXIF будет потерян."""
    with Image.open(io.BytesIO(content)) as img:
        ext = {"JPEG": ".jpg", "PNG": ".png", "HEIC": ".heic", "WEBP": ".webp"}.get(img.format or "", ".img")
    folder = Path(upload_dir) / f"visit_{visit_id}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(content)
    return path


def process_photo(
    db: Session,
    visit: Visit,
    content: bytes,
    received_at: datetime,
    *,
    upload_dir: str | Path,
    eps_gps_m: float = 100.0,
    dup_threshold: int = 5,
    p_low: float = 0.25,
    p_high: float = 0.75,
    weights: dict[str, float] | None = None,
    detector: Detector | None = None,
) -> Photo:
    store = visit.store
    path = save_original(content, upload_dir, visit.id)
    meta = read_metadata(content)
    phash = compute_phash(content)

    ctx = VisitContext(
        store_lat=store.lat, store_lon=store.lon, radius_m=store.radius_m, tz=store.tz,
        window_start=aware(visit.window_start), window_end=aware(visit.window_end),
        received_at=aware(received_at),
    )

    known = list(db.scalars(select(Photo.phash)))
    prev = [PhotoMeta.from_dict(p.meta) for p in visit.photos]
    points = []
    for m in prev + [meta]:
        t, _ = capture_time_utc(m, store.tz)
        if t is not None and m.gps_lat is not None and m.gps_lon is not None:
            points.append((t, m.gps_lat, m.gps_lon))

    checks = run_all_checks(meta, ctx, phash, known, prev, points, eps_gps_m, dup_threshold)

    detections, compl = [], None
    if detector is not None and visit.required_skus:
        dets = detector.detect(path)
        compl, counts, missing = completeness(visit.required_skus, dets)
        detections = [asdict(d) for d in dets]

    feats = dec.build_features(checks, meta.has_exif, compl)
    p = dec.probability(feats, weights)
    photo = Photo(
        visit_id=visit.id, path=str(path), phash=phash, received_at=received_at,
        meta=meta.to_dict(), checks={k: asdict(v) for k, v in checks.items()},
        detections=detections, completeness=compl, probability=p,
        decision=dec.decide(p, p_low, p_high), reasons=dec.reasons(feats, weights),
    )
    db.add(photo)
    db.commit()
    return photo
