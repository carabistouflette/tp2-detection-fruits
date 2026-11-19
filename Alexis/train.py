"""Fine-tune YOLO26n on the training split from TP1."""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prolonger", action="store_true",
                        help="Ajoute dix époques à partir du dernier checkpoint de la première phase.")
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.nnpack.set_flags(False)
    common = dict(data=str(ROOT / "dataset/data.yaml"), imgsz=640, batch=8,
                  project=str(ROOT / "runs"), device=args.device, workers=0, seed=42, plots=True)
    if args.prolonger:
        weights = (ROOT / "runs/fruits_detection/weights/last.pt"
                   if args.weights == "yolo26n.pt" else args.weights)
        YOLO(str(weights)).train(
            **common, epochs=10, name="fruits_prolongation", optimizer="AdamW",
            momentum=0.9, lr0=0.0003, lrf=0.1, warmup_epochs=0, mosaic=0, close_mosaic=0,
        )
    else:
        YOLO(args.weights).train(**common, epochs=20, name="fruits_detection")


if __name__ == "__main__":
    main()
