"""Compare the two detectors on their own test splits and a held-out intersection."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml
from ultralytics import YOLO

from evaluate import ROOT, selected_weights

ROBIN = ROOT.parent / "Robin"
SPLITS = ("train", "val", "test")
OUT = ROOT.parent / "comparaison"


def names_for(dataset):
    data = yaml.safe_load((dataset / "data.yaml").read_text())
    return {int(key): name.lower() for key, name in data["names"].items()}


def inventory(dataset):
    return {split: {path.name: path for path in (dataset / "images" / split).glob("*.jpg")}
            for split in SPLITS}


def digest(path):
    return hashlib.sha256(path.read_bytes()).digest()


def connected_to_seen(seen):
    with (ROOT / "images_liees.csv").open() as source:
        pairs = [(f"image{row['image_a']}.jpg", f"image{row['image_b']}.jpg")
                 for row in csv.DictReader(source)]
    expanded = set(seen)
    while True:
        related = {name for pair in pairs if expanded.intersection(pair) for name in pair}
        if related <= expanded:
            return expanded
        expanded.update(related)


def annotations(dataset, split, filename):
    path = dataset / "labels" / split / Path(filename).with_suffix(".txt")
    names = names_for(dataset)
    return [(names[int(fields[0])], tuple(map(float, fields[1:])))
            for line in path.read_text().splitlines() if (fields := line.split())]


def evaluate(model, data, device, tag, plots=False):
    validation = model.val(data=str(data), split="test", imgsz=640, batch=8, workers=0,
                           device=device, plots=plots, project=str(OUT), name=tag,
                           verbose=False, exist_ok=True)
    box = validation.box
    p, r = float(box.mp), float(box.mr)
    return {"precision": p, "rappel": r, "f1": 2*p*r/max(1e-12, p+r),
            "map50": float(box.map50), "map75": float(box.map75), "map50_95": float(box.map),
            "classes": {model.names[int(cls)].lower(): {
                "objets": int(validation.nt_per_class[cls]),
                "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]),
            } for i, cls in enumerate(box.ap_class_index)}}


def common_yaml(folder, samples, reference, names):
    ids = {name.lower(): idx for idx, name in names.items()}
    for subdirectory in ("images/test", "labels/test"):
        (folder / subdirectory).mkdir(parents=True, exist_ok=True)
    for name, photo in samples.items():
        (folder / "images/test" / name).symlink_to(photo)
        rows = (f"{ids[cls]} " + " ".join(map(str, box))
                for cls, box in annotations(reference, "test", name))
        (folder / "labels/test" / Path(name).with_suffix(".txt")).write_text(
            "".join(row + "\n" for row in rows))
    config = folder / "data.yaml"
    config.write_text(yaml.safe_dump({"path": str(folder), **dict.fromkeys(SPLITS, "images/test"),
                                      "names": dict(names)}))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.nnpack.set_flags(False)
    ours, robin = ROOT / "dataset", ROBIN / "dataset"
    left, right = inventory(ours), inventory(robin)
    seen = {name for dataset in (left, right) for split in ("train", "val") for name in dataset[split]}
    seen_hashes = {digest(p) for inv in (left, right) for split in ("train", "val")
                   for p in inv[split].values()}
    related = connected_to_seen(seen)
    shared = left["test"].keys() & right["test"].keys()
    samples = {name: left["test"][name] for name in sorted(shared - related)
               if digest(left["test"][name]) == digest(right["test"][name])
               and digest(left["test"][name]) not in seen_hashes}
    if not samples:
        raise ValueError("Aucune image commune hors train et validation.")
    subsets = {"tp2": ours, "robin": robin}
    models = {
        "tp2": YOLO(str(selected_weights())),
        "robin": YOLO(str(ROBIN / "runs/fruits_detection/weights/best.pt")),
    }
    yolo11 = ROBIN / "runs/fruits_yolo11n/weights/best.pt"
    if yolo11.is_file():
        models["robin_yolo11"] = YOLO(str(yolo11))
    summary = {
        "recouvrement": {s: {t: len(left[s].keys() & right[t].keys()) for t in SPLITS} for s in SPLITS},
        "images_test_communes_initiales": len(shared),
        "images_communes_evaluees": list(samples),
        "annotations_communes": {key: dict(Counter(cls for name in samples
                                                 for cls, _ in annotations(ds, "test", name)))
                                for key, ds in subsets.items()},
        "tests_origine": {},
        "test_commun": {},
    }
    OUT.mkdir(exist_ok=True)
    for key in models:
        ds = ours if key == "tp2" else robin
        summary["tests_origine"][key] = {
            "images": len(inventory(ds)["test"]),
            **evaluate(models[key], ds / "data.yaml", args.device, f"origine_{key}", plots=key != "tp2"),
        }
    with TemporaryDirectory(prefix="tp2-comparaison-") as temporary:
        robin_seen = set(right["train"]) | set(right["val"])
        robin_hashes = {digest(right[split][name]) for split in ("train", "val")
                        for name in right[split]}
        related_to_robin = connected_to_seen(robin_seen)
        robin_clean = {name: photo for name, photo in right["test"].items()
                       if name not in related_to_robin and digest(photo) not in robin_hashes}
        if robin_clean.keys() != right["test"].keys():
            for tag, model in models.items():
                if tag == "tp2":
                    continue
                clean_tag = f"{tag}_sans_variantes"
                data = common_yaml(Path(temporary) / clean_tag, robin_clean, robin, model.names)
                summary[clean_tag] = {
                    "images": len(robin_clean),
                    "images_ecartees": sorted(right["test"].keys() - robin_clean.keys()),
                    **evaluate(model, data, args.device, clean_tag),
                }
        for reference_name, reference in subsets.items():
            results = {}
            for model_name, model in models.items():
                tag = f"commun_{reference_name}_{model_name}"
                data = common_yaml(Path(temporary) / tag, samples, reference, model.names)
                results[model_name] = evaluate(model, data, args.device, tag)
            summary["test_commun"][reference_name] = results
    (OUT / "resultats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
