"""Compare saved detections with the test annotations at confidence 0.25."""
import csv
from collections import defaultdict

from evaluate import ROOT

CLASSES = ("red_apple", "green_apple", "banana", "orange")


def intersection_over_union(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


def labels(path):
    objects = []
    for line in path.read_text().splitlines():
        cls, x, y, w, h = map(float, line.split())
        objects.append((CLASSES[int(cls)], (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return objects


def diagnose(truth, predictions):
    free_truth = set(range(len(truth)))
    free_predictions = set(range(len(predictions)))
    pairs = [(intersection_over_union(t[1], p[1]), i, j)
             for i, t in enumerate(truth) for j, p in enumerate(predictions)]
    for iou, i, j in sorted(pairs, reverse=True):
        if iou < 0.10:
            break
        if i not in free_truth or j not in free_predictions:
            continue
        free_truth.remove(i)
        free_predictions.remove(j)
        actual, predicted = truth[i][0], predictions[j][0]
        if iou < 0.50:
            category = "boite_imprecise"
        elif actual != predicted:
            category = "classe_confondue"
        else:
            category = "correct"
        yield category, actual, predicted, iou, predictions[j][2], i + 1, j + 1
    for i in sorted(free_truth):
        yield "faux_negatif", truth[i][0], "", "", "", i + 1, ""
    for j in sorted(free_predictions):
        yield "faux_positif", "", predictions[j][0], "", predictions[j][2], "", j + 1


def main():
    predictions = defaultdict(list)
    folder = ROOT / "predictions/YOLO_Predictions"
    with (folder / "detections.csv").open() as stream:
        for row in csv.DictReader(stream):
            box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            predictions[row["image"]].append((row["classe"], box, float(row["confiance"])))
    with (folder / "analyse_erreurs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "cas", "reference", "prediction", "iou", "confiance",
                         "index_reference", "index_prediction"])
        for path in sorted((ROOT / "dataset/labels/test").glob("*.txt")):
            image = path.stem + ".jpg"
            writer.writerows((image, *row) for row in diagnose(labels(path), predictions[image]))


if __name__ == "__main__":
    main()
