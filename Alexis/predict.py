"""Draw detections and save their normalized coordinates for inspection."""
import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO

from evaluate import ROOT, selected_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/images/test")
    parser.add_argument("--weights", type=Path, default=selected_weights())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="YOLO_Predictions")
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.nnpack.set_flags(False)
    model = YOLO(str(args.weights))
    root = ROOT / "predictions"
    output = root / args.name
    output.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        source=str(args.source), conf=0.25, imgsz=640, save=True, stream=True,
        project=str(root), name=args.name, exist_ok=True, device=args.device,
    )
    with (output / "detections.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "classe", "confiance", "x1", "y1", "x2", "y2"])
        for result in results:
            for cls, score, coords in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                          result.boxes.xyxyn.tolist()):
                writer.writerow([Path(result.path).name, model.names[int(cls)], score, *coords])


if __name__ == "__main__":
    main()
