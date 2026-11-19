"""Select the best checkpoint across the two phases using validation mAP."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASES = (("fruits_detection", 20), ("fruits_prolongation", 10))
MAP = "metrics/mAP50-95(B)"


def read_history(root=ROOT):
    history = []
    offset = 0
    previous_time = 0.0
    for folder, epochs in PHASES:
        with (root / "runs" / folder / "results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != epochs:
            raise RuntimeError(f"{folder} : {len(rows)} époques sur {epochs}")
        for row in rows:
            row["epoch"] = str(offset + int(row["epoch"]))
            row["time"] = str(previous_time + float(row["time"]))
            row["phase"] = folder
        history.extend(rows)
        offset += epochs
        previous_time = float(history[-1]["time"])
    return history


def main():
    history = read_history()
    best = max(history, key=lambda row: (float(row[MAP]), int(row["epoch"])))
    weights = Path("runs") / best["phase"] / "weights/best.pt"
    if not (ROOT / weights).is_file():
        raise FileNotFoundError(ROOT / weights)
    selection = {"poids": str(weights), "epoque": int(best["epoch"]),
                 "map50_95_validation": float(best[MAP])}
    (ROOT / "modele_retenu.json").write_text(json.dumps(selection, indent=2) + "\n")
    with (ROOT / "runs/entrainement_30_epochs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(selection)


if __name__ == "__main__":
    main()
