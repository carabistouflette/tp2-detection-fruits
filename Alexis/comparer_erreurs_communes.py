"""Compare paired detection outcomes on the same test images."""
import csv
import json
from collections import Counter

from analyser_erreurs import intersection_over_union
from auditer_annotations import assign, read_labels
from comparer_confusions import predictions
from comparer_robin import OUT, ROBIN, ROOT, names_for


def nearest(box, others):
    return max((intersection_over_union(box, other) for _, other in others), default=0.0)


def outcome(reference, boxes, match):
    label, box = reference
    if match is not None:
        index, iou = match
        predicted = boxes[index][0]
        return ("correct" if predicted == label else "classe", predicted, iou)
    iou = nearest(box, boxes)
    return ("boite_imprecise" if 0.10 <= iou < 0.50 else
            "concurrence" if iou >= 0.50 else "manque", "", iou)


def write_csv(name, columns, data):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(data)


def main():
    images = json.loads((OUT/"resultats.json").read_text())["images_communes_evaluees"]
    models = [predictions(ROOT/"predictions/YOLO_Predictions/detections.csv"),
              predictions(ROBIN/"predictions/Robin_YOLO26/detections.csv")]
    objects, extras, summary = [], [], {}
    for tag, root in (("Alexis", ROOT/"dataset"), ("Robin", ROBIN/"dataset")):
        names = names_for(root)
        other_root = ROBIN/"dataset" if tag == "Alexis" else ROOT/"dataset"
        other_names = names_for(other_root)
        counts, surplus = Counter(), Counter()
        for name in images:
            reference, _ = read_labels(root, "test", name, names)
            alternative, _ = read_labels(other_root, "test", name, other_names)
            boxes = [model[name] for model in models]
            paired = [assign(reference, group) for group in boxes]
            by_object = [{j: (i, overlap) for i, (j, overlap) in side.items()} for side in paired]
            for j, target in enumerate(reference):
                a, b = [outcome(target, group, side.get(j)) for group, side in zip(boxes, by_object)]
                key = ("deux_corrects" if a[0] == b[0] == "correct" else
                       "alexis_seul_correct" if a[0] == "correct" else
                       "robin_seul_correct" if b[0] == "correct" else "deux_erreurs")
                counts[key] += 1
                objects.append((tag, name, j+1, target[0], *target[1], *a, *b, key))
            unmatched = [[box for i, box in enumerate(group) if i not in hits]
                         for group, hits in zip(boxes, paired)]
            shared = assign(unmatched[1], unmatched[0])
            reverse = {j: i for i, (j, _) in shared.items()}
            surplus["alexis"] += len(unmatched[0])
            surplus["robin"] += len(unmatched[1])
            surplus["paires_memes_regions"] += len(shared)
            for i, (group, hits) in enumerate(zip(unmatched, (shared, reverse))):
                for j, (label, box) in enumerate(group):
                    alt_iou = nearest(box, [candidate for candidate in alternative if candidate[0] == label])
                    extras.append((tag, name, ("Alexis", "Robin")[i], label, *box,
                                   "oui" if j in hits else "non", nearest(box, reference), alt_iou))
        summary[tag] = {"objets": dict(counts), "boites_en_trop": dict(surplus)}
    output_columns = ["statut", "classe_predite", "iou"]
    write_csv("erreurs_appariees.csv", ["reference", "image", "objet", "classe", "x1", "y1", "x2", "y2",
              *(f"{name}_{key}" for name in ("alexis", "robin") for key in output_columns), "bilan"], objects)
    write_csv("boites_en_trop_appariees.csv", ["reference", "image", "modele", "classe", "x1", "y1", "x2", "y2",
              "region_partagee", "iou_meilleure_reference", "iou_autre_reference_meme_classe"], extras)
    (OUT/"erreurs_communes.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
