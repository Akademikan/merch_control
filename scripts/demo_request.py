"""Демо: создаёт магазин и визит, отправляет 4 тестовых фото и печатает решения.

Запуск (API должен работать):  python scripts/demo_request.py
Нужен piexif:  pip install -r requirements-dev.txt
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.helpers import make_jpeg  # noqa: E402

API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
STORE = (59.9311, 30.3609)

now = datetime.now(timezone.utc).replace(microsecond=0)
start, end = now - timedelta(minutes=30), now + timedelta(minutes=30)
local = lambda dt: (dt + timedelta(hours=3)).replace(tzinfo=None)  # Москва, UTC+3

with httpx.Client(base_url=API, timeout=30) as c:
    sid = c.post("/stores", json={"name": "Демо-магазин", "lat": STORE[0], "lon": STORE[1], "radius_m": 60}).json()["id"]
    vid = c.post("/visits", json={"store_id": sid, "merchandiser": "m-demo",
                                  "window_start": start.isoformat(), "window_end": end.isoformat()}).json()["id"]
    cases = {
        "Нормальное фото": make_jpeg(seed=11, lat=STORE[0], lon=STORE[1], taken_local=local(now), gps_utc=now),
        "Другой город, вчера": make_jpeg(seed=12, lat=55.7558, lon=37.6173,
                                         taken_local=local(now - timedelta(days=1)), gps_utc=now - timedelta(days=1)),
        "EXIF стёрт (сжатое фото)": make_jpeg(seed=13, with_exif=False),
        "Повтор первого фото": make_jpeg(seed=11, lat=STORE[0], lon=STORE[1], taken_local=local(now), gps_utc=now),
    }
    for name, jpg in cases.items():
        r = c.post(f"/visits/{vid}/photos", files={"file": ("demo.jpg", jpg, "image/jpeg")},
                   data={"received_at": now.isoformat()}).json()
        print(f"{name:28s} -> {r['decision']:7s} p={r['probability']}  {r['reasons']}")
    print("\nИтог по визиту:", c.get(f"/visits/{vid}").json()["status"])
