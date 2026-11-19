"""Audit every Robin annotation and compare boxes on the same photos."""
import csv
import json
import math
from statistics import median
from collections import Counter
from pathlib import Path

from PIL import Image
from scipy.optimize import linear_sum_assignment

from analyser_erreurs import intersection_over_union
from comparer_robin import OUT, ROBIN, ROOT, SPLITS, digest, inventory, names_for


def read_labels(dataset, split, photo, names):
    labels = dataset / "labels" / split / Path(photo).with_suffix(".txt")
    if not labels.is_file():
        return [], ["fichier_absent"]
    flags = []
    boxes = []
    for number, line in enumerate(labels.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            identifier, *coordinates = line.split()
            cls = int(identifier)
            x, y, w, h = map(float, coordinates)
            if cls not in names or not all(map(math.isfinite, (x, y, w, h))):
                raise ValueError
        except ValueError:
            flags.append(f"ligne_{number}_invalide")
            continue
        xyxy = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        if w <= 0 or h <= 0 or min(xyxy) < -1e-6 or max(xyxy) > 1 + 1e-6:
            flags.append(f"ligne_{number}_hors_limites")
            continue
        box = (names[cls], xyxy)
        if box in boxes:
            flags.append(f"ligne_{number}_doublon")
        boxes.append(box)
    return boxes, flags


def flatten(items):
    return {name: (split, path) for split in SPLITS for name, path in items[split].items()}


def tally(boxes, counts):
    counts.update(cls for cls, _ in boxes)


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def assign(reference, boxes):
    if not reference or not boxes:
        return {}
    overlaps = [[intersection_over_union(a, b) for _, b in reference] for _, a in boxes]
    # Maximize the number of viable matches before choosing the tightest overlaps.
    weights = [[1 + iou / (len(boxes) + 1) if iou >= 0.5 else 0 for iou in row] for row in overlaps]
    left, right = linear_sum_assignment(weights, maximize=True)
    return {i: (j, overlaps[i][j]) for i, j in zip(left, right) if overlaps[i][j] >= 0.5}


def main():
    alexis, robin = ROOT / "dataset", ROBIN / "dataset"
    ref_names, robin_names = names_for(alexis), names_for(robin)
    ref_inventory, robin_inventory = flatten(inventory(alexis)), flatten(inventory(robin))
    OUT.mkdir(exist_ok=True)
    counts = Counter()
    per_split = {split: Counter() for split in SPLITS}
    per_class_pair = {cls: Counter() for cls in robin_names.values()}
    per_image, per_box = [], []
    overlap_values = []
    areas = {"robin": [], "alexis": []}
    for name, (split, path) in sorted(robin_inventory.items()):
        with Image.open(path) as image:
            image.verify()
        actual, flags = read_labels(robin, split, name, robin_names)
        tally(actual, per_split[split])
        counts["images"] += 1
        counts["boites_valides"] += len(actual)
        counts["images_sans_boite"] += not actual
        counts["images_1_boite"] += len(actual) == 1
        counts["images_plusieurs_boites"] += len(actual) > 1
        counts["images_anomalie_format"] += bool(flags)
        previous = ref_inventory.get(name)
        identical = previous is not None and digest(path) == digest(previous[1])
        reference, _ = read_labels(alexis, previous[0], name, ref_names) if identical else ([], [])
        pairs = assign(reference, actual)
        overlap_values.extend(overlap for _, overlap in pairs.values())
        paired_ref = {j for j, _ in pairs.values()}
        if identical:
            areas["robin"].extend(box_area(box) for _, box in actual)
            areas["alexis"].extend(box_area(box) for _, box in reference)
            counts["images_communes"] += 1
            counts["boites_robin_images_communes"] += len(actual)
            counts["boites_alexis_images_communes"] += len(reference)
            counts["robin_non_appariees"] += len(actual)-len(pairs)
            counts["alexis_non_appariees"] += len(reference)-len(pairs)
            counts["images_plus_boites_alexis"] += len(reference) > len(actual)
            counts["images_plus_boites_robin"] += len(actual) > len(reference)
            counts["images_effectif_identique"] += len(actual) == len(reference)
        image_counts = Counter()
        for index, (cls, box) in enumerate(actual):
            iou = None
            if index in pairs:
                j, iou = pairs[index]
                candidate = reference[j][0]
                status = "appariee_meme_classe" if cls == candidate else "appariee_classe_differente"
                per_class_pair[cls][candidate] += 1
            else:
                j, candidate = None, None
                status = "robin_non_appariee" if identical else "photo_hors_ref"
            image_counts[status] += 1
            counts[status] += 1
            per_box.append((name, split, "Robin", index+1, cls, *box, status,
                            j+1 if j is not None else "", candidate, iou))
        for index, (cls, box) in enumerate(reference):
            if index not in paired_ref:
                per_box.append((name, split, "Alexis", index+1, cls, *box, "alexis_non_appariee", "", "", ""))
        per_image.append((name, split, previous[0] if identical else "", len(actual),
                          len(reference) if identical else "", len(pairs),
                          image_counts["appariee_classe_differente"], ";".join(flags)))
    for folder in SPLITS:
        expected = {Path(name).with_suffix(".txt").name for name in inventory(robin)[folder]}
        existing = {path.name for path in (robin/"labels"/folder).glob("*.txt")}
        counts["fichiers_labels_sans_image"] += len(existing - expected)
    files = (
        ("audit_par_image.csv", ["image", "split_robin", "split_alexis", "n_robin", "n_alexis",
                                 "appariees", "classes_differentes", "anomalies_format"], per_image),
        ("audit_par_boite.csv", ["image", "split_robin", "origine", "index", "classe",
                                 "x1", "y1", "x2", "y2", "statut", "index_alexis",
                                 "classe_alexis", "iou"], per_box),
    )
    for filename, header, rows in files:
        with (OUT / filename).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    summary = {"totaux": counts, "classes_par_split_robin": per_split,
               "iou_mediane_appariees": median(overlap_values) if overlap_values else None,
               "appariees_iou_inferieure_0_75": sum(overlap < 0.75 for overlap in overlap_values),
               "surface_mediane_relative": {key: median(values) for key, values in areas.items() if values},
               "accord_classes": per_class_pair}
    (OUT / "audit_annotations.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
