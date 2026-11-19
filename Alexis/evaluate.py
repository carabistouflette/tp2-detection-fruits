"""Evaluate the validation-selected model on the TP1 test split."""
import argparse
import csv
import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import smooth

ROOT = Path(__file__).resolve().parent


def selected_weights():
    metadata = ROOT / "modele_retenu.json"
    if metadata.is_file():
        return ROOT / json.loads(metadata.read_text())["poids"]
    return ROOT / "runs/fruits_detection/weights/best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=selected_weights())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.nnpack.set_flags(False)
    model = YOLO(str(args.weights))
    result = model.val(
        data=str(ROOT / "dataset/data.yaml"), split="test", imgsz=640, batch=8,
        plots=True, project=str(ROOT / "evaluation_results"), name="YOLO_Test",
        device=args.device, workers=0, exist_ok=True,
    )
    box = result.box
    precision, recall = float(box.mp), float(box.mr)
    f1_index = smooth(box.f1_curve.mean(0), 0.1).argmax()
    summary = dict(
        images=len(list((ROOT / "dataset/images/test").glob("*.jpg"))),
        objects=int(result.nt_per_class.sum()),
        precision=precision, recall=recall,
        f1=2 * precision * recall / max(1e-12, precision + recall),
        confiance_point_ultralytics=float(box.px[f1_index]),
        map50=float(box.map50), map75=float(box.map75), map50_95=float(box.map),
    )
    output = Path(result.save_dir)
    (output / "metriques.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "classes.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["classe", "objets", "precision", "rappel", "f1", "ap50", "ap50_95"])
        for index, cls in enumerate(box.ap_class_index):
            p, r, ap50, ap = box.class_result(index)
            writer.writerow([model.names[int(cls)], int(result.nt_per_class[cls]),
                             p, r, float(box.f1[index]), ap50, ap])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
