"""Обучение детектора товаров (YOLO, Ultralytics).

Подготовка данных (CVAT / Label Studio / Roboflow -> экспорт в формате YOLO):
  data/yolo/images/{train,val,test}/*.jpg
  data/yolo/labels/{train,val,test}/*.txt
  data/yolo/data.yaml   (пример см. в README)
Делите выборку ПО МАГАЗИНАМ, а не по случайным кадрам: фото одной полки не должны быть и в train, и в test.

Запуск:  python scripts/train_detector.py --data data/yolo/data.yaml --model yolov8s.pt --epochs 80
Результат копируется в data/detector.pt — его подхватит API.
Метрики (precision, recall, mAP@0.5) печатаются для val и test.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=960)  # на полках мелкие товары — размер лучше увеличить
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)  # "0" для первой видеокарты, "cpu" для процессора
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, project="runs", name="sku")
    best = Path(model.trainer.best)
    for split in ("val", "test"):
        m = YOLO(str(best)).val(data=a.data, split=split, imgsz=a.imgsz, device=a.device)
        print(f"[{split}] P={m.box.mp:.3f} R={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
    Path("data").mkdir(exist_ok=True)
    shutil.copy(best, "data/detector.pt")
    print("Сохранено: data/detector.pt")


if __name__ == "__main__":
    main()
