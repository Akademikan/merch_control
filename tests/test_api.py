from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.helpers import make_jpeg

STORE = (59.9311, 30.3609)


def client(tmp_path):
    cfg = Settings(database_url=f"sqlite:///{tmp_path}/t.db", upload_dir=str(tmp_path / "up"),
                   weights_path=None, detector_weights=None)
    return TestClient(create_app(cfg))


def setup_visit(c):
    sid = c.post("/stores", json={"name": "Магазин 1", "lat": STORE[0], "lon": STORE[1], "radius_m": 60}).json()["id"]
    return c.post("/visits", json={
        "store_id": sid, "merchandiser": "m-001",
        "window_start": "2026-11-05T07:00:00+00:00", "window_end": "2026-11-05T08:00:00+00:00",
    }).json()["id"]


def upload(c, vid, jpg, received="2026-11-05T07:40:00+00:00"):
    return c.post(f"/visits/{vid}/photos", files={"file": ("a.jpg", jpg, "image/jpeg")}, data={"received_at": received})


def test_end_to_end(tmp_path):
    c = client(tmp_path)
    vid = setup_visit(c)

    good = make_jpeg(seed=1, lat=STORE[0], lon=STORE[1], taken_local=datetime(2026, 11, 5, 10, 35, 0),
                     gps_utc=datetime(2026, 11, 5, 7, 35, 0))
    r = upload(c, vid, good)
    assert r.status_code == 200 and r.json()["decision"] == "accept"

    # то же фото второй раз — дубликат
    r = upload(c, vid, good)
    assert r.json()["decision"] == "reject"
    assert "фото похоже на ранее загруженное" in r.json()["reasons"]

    # фото без EXIF (как после сжатия в Telegram) — ручная проверка
    r = upload(c, vid, make_jpeg(seed=2, with_exif=False))
    assert r.json()["decision"] == "manual"

    rep = c.get(f"/visits/{vid}").json()
    assert rep["status"] == "reject" and len(rep["photos"]) == 3

    assert c.post("/photos/1/review", params={"supervisor_decision": "accept"}).json() == {"ok": True}


def test_errors(tmp_path):
    c = client(tmp_path)
    assert upload(c, 999, make_jpeg()).status_code == 404
    vid = setup_visit(c)
    assert upload(c, vid, b"not an image").status_code == 400
    assert c.post("/visits", json={"store_id": 42, "merchandiser": "x",
                                   "window_start": "2026-11-05T07:00:00+00:00",
                                   "window_end": "2026-11-05T08:00:00+00:00"}).status_code == 404
