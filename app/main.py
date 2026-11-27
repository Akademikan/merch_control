"""FastAPI-приложение. Запуск: uvicorn app.main:app --reload"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import Settings
from .db import Photo, Store, Visit, make_session_factory
from .services import decision as dec
from .services.pipeline import process_photo


class StoreIn(BaseModel):
    name: str
    lat: float
    lon: float
    radius_m: float = 60.0
    tz: str = "Europe/Moscow"


class VisitIn(BaseModel):
    store_id: int
    merchandiser: str
    window_start: datetime
    window_end: datetime
    required_skus: dict[str, int] = {}  # {"sku": минимум фейсингов}


def photo_view(p: Photo) -> dict:
    return {
        "id": p.id, "decision": p.decision, "probability": round(p.probability, 3),
        "reasons": p.reasons, "completeness": p.completeness,
        "checks": {k: {"status": v["status"], "detail": v["detail"]} for k, v in p.checks.items()},
        "detections": p.detections,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    Path(cfg.upload_dir).mkdir(parents=True, exist_ok=True)
    Session_ = make_session_factory(cfg.database_url)
    weights = dec.load_weights(cfg.weights_path)

    detector = None
    if cfg.detector_weights and Path(cfg.detector_weights).exists():
        try:
            from .services.detector import Detector

            detector = Detector(cfg.detector_weights)
        except Exception as exc:  # нет torch/ultralytics — работаем только по метаданным
            print(f"[warn] детектор отключён: {exc}")

    app = FastAPI(title="МерчКонтроль", version="0.1")

    def get_db():
        with Session_() as db:
            yield db

    @app.post("/stores")
    def add_store(body: StoreIn, db: Session = Depends(get_db)):
        s = Store(**body.model_dump())
        db.add(s)
        db.commit()
        return {"id": s.id}

    @app.post("/visits")
    def add_visit(body: VisitIn, db: Session = Depends(get_db)):
        if not db.get(Store, body.store_id):
            raise HTTPException(404, "магазин не найден")
        v = Visit(**body.model_dump())
        db.add(v)
        db.commit()
        return {"id": v.id}

    @app.post("/visits/{visit_id}/photos")
    async def upload_photo(
        visit_id: int,
        file: UploadFile = File(...),
        received_at: str | None = Form(None),
        db: Session = Depends(get_db),
    ):
        visit = db.get(Visit, visit_id)
        if not visit:
            raise HTTPException(404, "визит не найден")
        content = await file.read()
        # Время получения лучше брать от источника, которому доверяем (сервер или Telegram).
        t = datetime.fromisoformat(received_at) if received_at else datetime.now(timezone.utc)
        try:
            photo = process_photo(
                db, visit, content, t, upload_dir=cfg.upload_dir, eps_gps_m=cfg.eps_gps_m,
                dup_threshold=cfg.dup_threshold, p_low=cfg.p_low, p_high=cfg.p_high,
                weights=weights, detector=detector,
            )
        except OSError:
            raise HTTPException(400, "файл не распознан как изображение")
        return photo_view(photo)

    @app.get("/visits/{visit_id}")
    def visit_report(visit_id: int, db: Session = Depends(get_db)):
        visit = db.get(Visit, visit_id)
        if not visit:
            raise HTTPException(404, "визит не найден")
        decisions = [p.decision for p in visit.photos]
        total = "reject" if "reject" in decisions else "manual" if "manual" in decisions else "accept"
        return {
            "visit_id": visit.id, "status": total if decisions else "empty",
            "photos": [photo_view(p) for p in visit.photos],
        }

    @app.post("/photos/{photo_id}/review")
    def review(photo_id: int, supervisor_decision: str, db: Session = Depends(get_db)):
        """Решение супервайзера сохраняется как разметка для обучения решающего модуля."""
        if supervisor_decision not in ("accept", "reject"):
            raise HTTPException(400, "accept или reject")
        p = db.get(Photo, photo_id)
        if not p:
            raise HTTPException(404, "фото не найдено")
        p.supervisor_decision = supervisor_decision
        db.commit()
        return {"ok": True}

    return app


app = create_app()
