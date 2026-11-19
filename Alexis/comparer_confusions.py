"""Build detection confusion matrices at conf 0.25 and IoU 0.50."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from auditer_annotations import assign, read_labels
from comparer_robin import OUT, ROBIN, ROOT, names_for

CLASSES = ("red_apple", "green_apple", "banana", "orange", "fond")


def predictions(path):
    by_image = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            coords = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            by_image[row["image"]].append((row["classe"].lower(), coords))
    return by_image


def measure(dataset, output, images=None):
    names = names_for(dataset)
    preds = predictions(output / "detections.csv")
    matrix = {ref: dict.fromkeys(CLASSES, 0) for ref in CLASSES}
    if images is None:
        images = sorted(path.name for path in (dataset / "images/test").glob("*.jpg"))
    for name in images:
        truth, _ = read_labels(dataset, "test", name, names)
        boxes = preds[name]
        pairs = assign(truth, boxes)
        related = {k for k, _ in pairs.values()}
        for i, (label, _) in enumerate(boxes):
            ref = truth[pairs[i][0]][0] if i in pairs else "fond"
            matrix[ref][label] += 1
        for j, (label, _) in enumerate(truth):
            if j not in related:
                matrix[label]["fond"] += 1
    valid = CLASSES[:-1]
    correct = sum(matrix[c][c] for c in valid)
    classes_confused = sum(matrix[a][b] for a in valid for b in valid if a != b)
    missing = sum(matrix[c]["fond"] for c in valid)
    extra = sum(matrix["fond"][c] for c in valid)
    fp, fn = extra + classes_confused, missing + classes_confused
    return {
        "images": len(images), "objets": correct + classes_confused + missing,
        "correct": correct, "classe_confondue": classes_confused,
        "sans_prediction": missing, "sans_annotation": extra,
        "accuracy_classe_boites_appariees": correct / max(1, correct + classes_confused),
        "precision_micro": correct / max(1, correct + fp),
        "rappel_micro": correct / max(1, correct + fn),
        "f1_micro": 2*correct / max(1, 2*correct + fp + fn),
        "matrice_lignes_reference": matrix,
    }


def main():
    ours, robin = ROOT/"dataset", ROBIN/"dataset"
    pred_dir = ROOT/"predictions"
    models = {
        "tp2": (ours, pred_dir/"YOLO_Predictions"),
        "robin": (robin, ROBIN/"predictions/Robin_YOLO26"),
        "robin_yolo11": (robin, ROBIN/"predictions/Robin_YOLO11"),
    }
    common = json.loads((OUT/"resultats.json").read_text())["images_communes_evaluees"]
    result = {}
    for tag, (dataset, output) in models.items():
        if not (output/"detections.csv").is_file():
            continue
        result[tag] = {"test_origine": measure(dataset, output),
                       "commun_alexis": measure(ours, output, common),
                       "commun_robin": measure(robin, output, common)}
        with (OUT/f"confusion_{tag}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["reference"] + list(CLASSES))
            for cls, values in result[tag]["test_origine"]["matrice_lignes_reference"].items():
                writer.writerow([cls, *values.values()])
    (OUT/"confusions.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
