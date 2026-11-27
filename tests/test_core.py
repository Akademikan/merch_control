from datetime import datetime, timedelta, timezone

import pytest

from app.services import decision as dec
from app.services.detector import Detection, completeness, iou, nms
from app.services.duplicates import compute_phash, hamming
from app.services.exif import read_metadata
from app.services.geo import haversine_m
from app.services.verify import (
    VisitContext, capture_time_utc, check_duplicate, check_in_window, check_location,
    check_series_consistency, check_software, check_speed, check_upload_delay, run_all_checks,
)
from tests.helpers import make_jpeg

STORE = (59.9311, 30.3609)
UTC = timezone.utc
# Визит 10:00-11:00 по Москве = 07:00-08:00 UTC
WIN_START = datetime(2026, 11, 5, 7, 0, tzinfo=UTC)
WIN_END = datetime(2026, 11, 5, 8, 0, tzinfo=UTC)


def ctx(received=None):
    return VisitContext(STORE[0], STORE[1], 60.0, "Europe/Moscow", WIN_START, WIN_END,
                        received or datetime(2026, 11, 5, 7, 40, tzinfo=UTC))


# ---------- геометрия ----------
def test_haversine_known_distance():
    d = haversine_m(55.7558, 37.6173, 59.9343, 30.3351)  # Москва - Санкт-Петербург
    assert 630_000 < d < 640_000


def test_haversine_zero_and_symmetry():
    assert haversine_m(*STORE, *STORE) == pytest.approx(0.0, abs=1e-6)
    assert haversine_m(1, 2, 3, 4) == pytest.approx(haversine_m(3, 4, 1, 2))


# ---------- EXIF ----------
def test_exif_roundtrip():
    jpg = make_jpeg(lat=59.9311, lon=30.3609, taken_local=datetime(2026, 11, 5, 10, 30, 5),
                    gps_utc=datetime(2026, 11, 5, 7, 30, 5), software="One UI", offset="+03:00")
    m = read_metadata(jpg)
    assert m.has_exif
    assert m.gps_lat == pytest.approx(59.9311, abs=1e-4)
    assert m.gps_lon == pytest.approx(30.3609, abs=1e-4)
    assert m.taken_local == datetime(2026, 11, 5, 10, 30, 5)
    assert m.gps_time_utc == datetime(2026, 11, 5, 7, 30, 5, tzinfo=UTC)
    assert (m.make, m.model, m.software, m.utc_offset) == ("Samsung", "SM-A546B", "One UI", "+03:00")


def test_exif_southern_western_hemisphere():
    m = read_metadata(make_jpeg(lat=-33.8688, lon=-70.6693))
    assert m.gps_lat == pytest.approx(-33.8688, abs=1e-4)
    assert m.gps_lon == pytest.approx(-70.6693, abs=1e-4)


def test_no_exif_does_not_crash():
    m = read_metadata(make_jpeg(with_exif=False))
    assert not m.has_exif and m.gps_lat is None and m.taken_local is None


def test_meta_dict_roundtrip():
    m = read_metadata(make_jpeg(lat=59.9, lon=30.3, taken_local=datetime(2026, 11, 5, 10, 0, 0),
                                gps_utc=datetime(2026, 11, 5, 7, 0, 0)))
    assert type(m).from_dict(m.to_dict()) == m


# ---------- время ----------
def test_capture_time_sources():
    gps = read_metadata(make_jpeg(gps_utc=datetime(2026, 11, 5, 7, 30, 0)))
    assert capture_time_utc(gps, "Europe/Moscow")[1] == "gps"
    off = read_metadata(make_jpeg(taken_local=datetime(2026, 11, 5, 10, 30, 0), offset="+03:00"))
    t, src = capture_time_utc(off, "Europe/Moscow")
    assert src == "exif+offset" and t == datetime(2026, 11, 5, 7, 30, tzinfo=UTC)
    local = read_metadata(make_jpeg(taken_local=datetime(2026, 11, 5, 10, 30, 0)))
    t, src = capture_time_utc(local, "Europe/Moscow")
    assert src == "exif+store_tz" and t == datetime(2026, 11, 5, 7, 30, tzinfo=UTC)


def test_in_window_ok_fail_unknown():
    ok = read_metadata(make_jpeg(taken_local=datetime(2026, 11, 5, 10, 30, 0)))
    assert check_in_window(ok, ctx()).status == "ok"
    yesterday = read_metadata(make_jpeg(taken_local=datetime(2026, 11, 4, 10, 30, 0)))
    r = check_in_window(yesterday, ctx())
    assert r.status == "fail" and r.value > 60
    assert check_in_window(read_metadata(make_jpeg(with_exif=False)), ctx()).status == "unknown"


def test_upload_delay():
    m = read_metadata(make_jpeg(taken_local=datetime(2026, 11, 5, 10, 30, 0)))  # 07:30 UTC
    assert check_upload_delay(m, ctx(datetime(2026, 11, 5, 7, 40, tzinfo=UTC))).status == "ok"
    assert check_upload_delay(m, ctx(datetime(2026, 11, 5, 14, 0, tzinfo=UTC))).status == "fail"
    # «снято в будущем» — часы телефона
    assert check_upload_delay(m, ctx(datetime(2026, 11, 5, 7, 0, tzinfo=UTC))).status == "fail"


# ---------- место ----------
def test_location_checks():
    near = read_metadata(make_jpeg(lat=STORE[0] + 0.0003, lon=STORE[1]))  # ~33 м
    assert check_location(near, ctx()).status == "ok"
    far = read_metadata(make_jpeg(lat=STORE[0] + 0.05, lon=STORE[1]))  # ~5.5 км
    r = check_location(far, ctx())
    assert r.status == "fail" and 5000 < r.value < 6000
    assert check_location(read_metadata(make_jpeg()), ctx()).status == "unknown"


def test_location_eps_tolerance():
    m = read_metadata(make_jpeg(lat=STORE[0] + 0.001, lon=STORE[1]))  # ~111 м
    assert check_location(m, ctx(), eps_gps_m=0).status == "fail"
    assert check_location(m, ctx(), eps_gps_m=100).status == "ok"


# ---------- прочее ----------
def test_software_and_devices():
    assert check_software(read_metadata(make_jpeg(software="Adobe Photoshop 25"))).status == "fail"
    assert check_software(read_metadata(make_jpeg(software="One UI 6.1"))).status == "ok"
    a = read_metadata(make_jpeg(make="Samsung", model="A"))
    b = read_metadata(make_jpeg(make="Apple", model="iPhone 14"))
    assert check_series_consistency([a, a]).status == "ok"
    assert check_series_consistency([a, b]).status == "fail"


def test_speed():
    t = datetime(2026, 11, 5, 7, 0, tzinfo=UTC)
    near = [(t, *STORE), (t + timedelta(minutes=5), STORE[0] + 0.0005, STORE[1])]
    assert check_speed(near).status == "ok"
    teleport = [(t, *STORE), (t + timedelta(minutes=5), 55.7558, 37.6173)]  # СПб -> Москва за 5 мин
    assert check_speed(teleport).status == "fail"
    assert check_speed(near[:1]).status == "unknown"


def test_phash_duplicates():
    a = make_jpeg(seed=1)
    b = make_jpeg(seed=2)
    assert hamming(compute_phash(a), compute_phash(a)) == 0
    assert hamming(compute_phash(a), compute_phash(b)) > 5
    # то же фото, но другие метаданные — всё равно дубликат
    a2 = make_jpeg(seed=1, lat=1.0, lon=1.0)
    assert check_duplicate(compute_phash(a2), [compute_phash(a)]).status == "fail"
    assert check_duplicate(compute_phash(b), [compute_phash(a)]).status == "ok"


# ---------- решающий модуль ----------
def _decision_for(jpg, known=(), received=None):
    meta = read_metadata(jpg)
    checks = run_all_checks(meta, ctx(received), compute_phash(jpg), list(known), [], [])
    f = dec.build_features(checks, meta.has_exif, None)
    p = dec.probability(f)
    return dec.decide(p), p, dec.reasons(f)


def test_clean_report_accepted():
    jpg = make_jpeg(lat=STORE[0], lon=STORE[1], taken_local=datetime(2026, 11, 5, 10, 35, 0),
                    gps_utc=datetime(2026, 11, 5, 7, 35, 0))
    d, p, reasons = _decision_for(jpg)
    assert d == "accept" and p < 0.25 and reasons == []


def test_stripped_exif_goes_to_manual():
    d, p, reasons = _decision_for(make_jpeg(with_exif=False))
    assert d == "manual"
    assert any("метаданных" in r for r in reasons)


def test_wrong_place_and_day_rejected():
    jpg = make_jpeg(lat=55.7558, lon=37.6173, taken_local=datetime(2026, 11, 3, 10, 35, 0),
                    gps_utc=datetime(2026, 11, 3, 7, 35, 0))
    d, p, reasons = _decision_for(jpg)
    assert d == "reject" and p > 0.75
    assert "фото сделано не в магазине" in reasons


def test_duplicate_rejected():
    jpg = make_jpeg(seed=7, lat=STORE[0], lon=STORE[1], taken_local=datetime(2026, 11, 5, 10, 35, 0),
                    gps_utc=datetime(2026, 11, 5, 7, 35, 0))
    d, p, reasons = _decision_for(jpg, known=[compute_phash(jpg)])
    assert d == "reject" and "фото похоже на ранее загруженное" in reasons


def test_missing_products_raise_probability():
    base = {k: 0.0 for k in dec.FEATURES}
    assert dec.probability({**base, "missing_share": 0.8}) > dec.probability(base)
    assert dec.decide(0.1) == "accept" and dec.decide(0.5) == "manual" and dec.decide(0.9) == "reject"


# ---------- детектор ----------
def test_iou_and_nms():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    dets = [Detection("cola", 0.9, (0, 0, 10, 10)), Detection("cola", 0.6, (1, 1, 11, 11)),
            Detection("fanta", 0.7, (1, 1, 11, 11))]
    kept = nms(dets, 0.5)
    assert [(d.label, d.conf) for d in kept] == [("cola", 0.9), ("fanta", 0.7)]


def test_completeness():
    found = [Detection("cola", 0.9, (0, 0, 1, 1)), Detection("cola", 0.8, (2, 0, 3, 1)),
             Detection("fanta", 0.7, (4, 0, 5, 1))]
    share, counts, missing = completeness({"cola": 2, "fanta": 1, "sprite": 1}, found)
    assert share == pytest.approx(2 / 3) and missing == ["sprite"] and counts == {"cola": 2, "fanta": 1}
    share, _, missing = completeness({"cola": 3}, found)  # фейсингов меньше минимума
    assert share == 0.0 and missing == ["cola"]
    assert completeness({}, found)[0] == 1.0
