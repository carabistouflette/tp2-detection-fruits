"""Entraîner YOLO11n avec le protocole de Robin."""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local_weights = ROOT / "yolo11n.pt"
    parser.add_argument("--weights", default=str(local_weights) if local_weights.is_file() else "yolo11n.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.nnpack.set_flags(False)
    YOLO(args.weights).train(data=str(ROOT / "dataset/data.yaml"), epochs=30, patience=5,
                            imgsz=640, batch=8, seed=42, device=args.device, workers=0,
                            project=str(ROOT / "runs"), name="fruits_yolo11n", plots=True)


if __name__ == "__main__":
    main()
