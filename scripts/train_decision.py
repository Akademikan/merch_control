"""Обучение решающего модуля (логистическая регрессия) на размеченных отчётах.

Источники разметки (одно из двух):
  --csv файл.csv   колонки = FEATURES из app/services/decision.py + label (1 — нарушение, 0 — норма)
  --db URL         берутся фото, у которых супервайзер проставил решение (accept -> 0, reject -> 1)

Результат: data/decision_weights.json (его подхватывает API) и рекомендуемые пороги p_low / p_high.
Пример:  python scripts/train_decision.py --csv labeled.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.decision import FEATURES  # noqa: E402


def load_csv(path: str):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    return X, y


def load_db(url: str):
    from sqlalchemy import select

    from app.db import Photo, make_session_factory
    from app.services.decision import build_features
    from app.services.verify import CheckResult

    X, y = [], []
    with make_session_factory(url)() as db:
        for p in db.scalars(select(Photo).where(Photo.supervisor_decision.is_not(None))):
            checks = {k: CheckResult(**v) for k, v in p.checks.items()}
            f = build_features(checks, p.meta.get("has_exif", False), p.completeness)
            X.append([f[k] for k in FEATURES])
            y.append(1 if p.supervisor_decision == "reject" else 0)
    return np.array(X), np.array(y)


def thresholds(y_true, p, recall_target=0.95, precision_target=0.95):
    """p_low: самый высокий порог, ниже которого пропускается не более 5% нарушений.
    p_high: самый низкий порог, выше которого не менее 95% отклонений — настоящие нарушения."""
    p_low, p_high = 0.0, 1.0
    for t in np.linspace(0.01, 0.99, 99):
        if (p[y_true == 1] < t).mean() <= 1 - recall_target if (y_true == 1).any() else False:
            p_low = t
    for t in np.linspace(0.99, 0.01, 99):
        sel = p > t
        if sel.sum() and y_true[sel].mean() >= precision_target:
            p_high = t
    if p_low > p_high:  # классы разделились чисто: зона ручной проверки не нужна, берём середину
        p_low = p_high = (p_low + p_high) / 2
    return float(p_low), float(p_high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--db")
    ap.add_argument("--out", default="data/decision_weights.json")
    ap.add_argument("--test-size", type=float, default=0.3)
    a = ap.parse_args()
    if not (a.csv or a.db):
        ap.error("нужен --csv или --db")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = load_csv(a.csv) if a.csv else load_db(a.db)
    print(f"Примеров: {len(y)}, нарушений: {int(y.sum())}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=a.test_size, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)[:, 1]
    print(f"ROC-AUC на тесте: {roc_auc_score(yte, p_te):.3f}")
    lo, hi = thresholds(yte, p_te)
    print(f"Рекомендуемые пороги: p_low={lo:.2f}, p_high={hi:.2f}")

    final = LogisticRegression(max_iter=1000).fit(X, y)  # итоговая модель — на всех данных
    weights = {"bias": float(final.intercept_[0]), **{k: float(w) for k, w in zip(FEATURES, final.coef_[0])}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Веса сохранены: {a.out}")


if __name__ == "__main__":
    main()
