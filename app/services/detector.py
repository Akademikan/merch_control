"""Распознавание товаров на фото.

Обёртка над детектором (по умолчанию Ultralytics YOLO; лицензия AGPL-3.0 — упомянуть в ВКР).
Модель импортируется лениво: проект запускается и без torch, тогда проверка товаров
просто пропускается, а решение принимается только по метаданным.

Классы модели = SKU. Для MVP достаточно 5-10 SKU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 в пикселях


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU = площадь пересечения / площадь объединения двух рамок."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Подавление дублей рамок одного класса (в YOLO уже есть, функция для разбора в тексте ВКР и тестов)."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: x.conf, reverse=True):
        if all(k.label != d.label or iou(k.box, d.box) < iou_thr for k in kept):
            kept.append(d)
    return kept


class Detector:
    def __init__(self, weights: str | Path, conf: float = 0.25, iou_thr: float = 0.5, device: str | None = None):
        from ultralytics import YOLO  # ленивый импорт

        self.model = YOLO(str(weights))
        self.conf, self.iou_thr, self.device = conf, iou_thr, device

    def detect(self, image_path: str | Path) -> list[Detection]:
        res = self.model.predict(str(image_path), conf=self.conf, iou=self.iou_thr, device=self.device, verbose=False)[0]
        out: list[Detection] = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(Detection(res.names[int(b.cls[0])], float(b.conf[0]), (x1, y1, x2, y2)))
        return out


def completeness(required: dict[str, int], found: list[Detection]) -> tuple[float, dict[str, int], list[str]]:
    """Полнота выкладки Compl = |S ∩ D| / |S|.

    required — чек-лист: SKU -> минимальное число фейсингов.
    Возвращает (Compl, число найденных фейсингов по SKU, список недостающих SKU).
    """
    if not required:
        return 1.0, {}, []
    counts: dict[str, int] = {}
    for d in found:
        counts[d.label] = counts.get(d.label, 0) + 1
    missing = [sku for sku, need in required.items() if counts.get(sku, 0) < need]
    return (len(required) - len(missing)) / len(required), counts, missing
