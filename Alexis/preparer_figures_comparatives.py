"""Generate the report tables and figures from the measured results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "comparaison"
FIG = OUT / "figures"
TAGS = ("tp2", "robin", "robin_yolo11")
NAMES = ("Alexis 26n", "Robin 26n", "Robin 11n")
RUNS = ("runs/entrainement_30_epochs.csv", "../Robin/runs/fruits_detection/results.csv",
        "../Robin/runs/fruits_yolo11n/results.csv")
AXES = ("Rouge", "Verte", "Banane", "Orange", "Fond")
MAP = "metrics/mAP50-95(B)"


def load(name):
    return json.loads((OUT / name).read_text())


def rows(lines):
    return "\n".join(" & ".join(map(str, row)) + r" \\" for row in lines)


def pct(number):
    return (f"{number*100:.1f}" + r"\,\%").replace(".", ",")


def macro(name, value):
    return rf"\newcommand{{\{name}}}{{{value}}}" + "\n"


def training():
    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.6), layout="constrained", sharey=True)
    colors = ("#217a82", "#493c90", "#c2593d")
    for i, (name, path, color) in enumerate(zip(NAMES, RUNS, colors)):
        with (ROOT/path).open() as source:
            reader = csv.DictReader(source)
            reader.fieldnames = [key.strip() for key in reader.fieldnames]
            history = [{"epoch": int(row["epoch"]), MAP: float(row[MAP])} for row in reader]
        best = max(history, key=lambda row: (row[MAP], row["epoch"]))
        epochs = [int(row["epoch"]) for row in history]
        axis = axes[0] if i == 0 else axes[1]
        axis.plot(epochs, [row[MAP]*100 for row in history], label=name, color=color, linewidth=1.5)
        axis.scatter(int(best["epoch"]), best[MAP]*100, s=18, color=color, zorder=2)
        summaries.append((name, "20 + 10" if i == 0 else "Continu",
                          f"{len(history)} / {int(best['epoch'])}", pct(best[MAP])))
    axes[0].axvline(20.5, color="0.6", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("mAP de 0,50 à 0,95 (%)")
    for axis, title in zip(axes, ("Validation Alexis", "Validation Robin")):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Époque")
        axis.set_xticks([1, 10, 20, 30])
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8, loc="lower right")
    fig.savefig(FIG/"validation_comparative.pdf")
    plt.close(fig)
    return summaries


def plot_confusions(confusion):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3), layout="constrained")
    maximum = max(val for tag in TAGS for row in confusion[tag]["test_origine"]["matrice_lignes_reference"].values()
                  for val in row.values())
    for axis, tag, name in zip(axes, TAGS, NAMES):
        matrix = confusion[tag]["test_origine"]["matrice_lignes_reference"]
        data = [list(row.values()) for row in matrix.values()]
        axis.imshow(data, cmap="Blues", vmin=0, vmax=maximum)
        for y, row in enumerate(data):
            for x, value in enumerate(row):
                axis.text(x, y, str(value) if value else "·", ha="center", va="center",
                          fontsize=9, color="white" if value > maximum * 0.56 else "#25344a")
        axis.set_xticks(range(5), AXES, rotation=35, ha="right", fontsize=8)
        axis.set_yticks(range(5), AXES, fontsize=8)
        axis.set_title(name, fontsize=10)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.savefig(FIG/"confusions_comparatives.pdf")
    plt.close(fig)


def main():
    scores = load("resultats.json")
    conf = load("confusions.json")
    audit = load("audit_annotations.json")
    native = scores["tests_origine"]
    common = scores["test_commun"]
    counts = audit["totaux"]
    areas = audit["surface_mediane_relative"]
    plot_confusions(conf)
    fit = training()
    audit_text = (
        f"Sur les 300 photos et {counts['boites_valides']} boîtes de Robin, les fichiers ont été "
        "contrôlés et les images relues. Aucun défaut de format. Sur les 298 photos identiques, "
        f"{counts['appariee_meme_classe']} boîtes s'apparient à IoU 0,50, toutes de même classe "
        f"(IoU médiane {audit['iou_mediane_appariees']:.2f}). "
        f"Il reste {counts['robin_non_appariees']} boîtes de Robin et "
        f"{counts['alexis_non_appariees']} d'Alexis sans correspondance. "
        f"La surface médiane d'une boîte vaut {pct(areas['robin'])} de l'image chez Robin "
        f"et {pct(areas['alexis'])} chez Alexis."
    ).replace(f"{audit['iou_mediane_appariees']:.2f}",
              f"{audit['iou_mediane_appariees']:.2f}".replace(".", ","))
    texts = [macro("AuditText", audit_text), macro("TrainRows", rows(fit))]
    native_rows, confusion_rows = [], []
    for key, label in zip(TAGS, NAMES):
        data = native[key]
        c = conf[key]["test_origine"]
        native_rows.append((label, data["images"], c["objets"],
                            *(pct(data[k]) for k in ("precision", "rappel", "map50", "map50_95"))))
        confusion_rows.append((label, *(c[k] for k in ("correct", "classe_confondue",
                                                       "sans_prediction", "sans_annotation")),
                               pct(c["accuracy_classe_boites_appariees"]), pct(c["f1_micro"])))
    texts.append(macro("NativeRows", rows(native_rows)))
    texts.append(macro("ConfusionRows", rows(confusion_rows)))
    classes = (("red_apple", "Pomme rouge"), ("green_apple", "Pomme verte"),
               ("banana", "Banane"), ("orange", "Orange"))
    texts.append(macro("ClassRows", rows([
        (label, f"{native['tp2']['classes'][cls]['objets']} / {native['robin']['classes'][cls]['objets']}",
         *(pct(native[tag]["classes"][cls]["ap50_95"]) for tag in TAGS))
        for cls, label in classes
    ])))
    texts.append(macro("CommonRows", rows([
        (label, sum(scores["annotations_communes"][ref].values()),
         *(pct(common[ref][tag]["map50_95"]) for tag in TAGS))
        for label, ref in (("Alexis, boîte par instance", "tp2"), ("Robin", "robin"))
    ])))
    texts.append(macro("CleanComment",
                       "Après retrait, sur 43 images, la mAP de 0,50 à 0,95 vaut "
                       f"{pct(scores['robin_sans_variantes']['map50_95'])} pour YOLO26n et "
                       f"{pct(scores['robin_yolo11_sans_variantes']['map50_95'])} pour YOLO11n."))
    paired = load("erreurs_communes.json")
    texts.append(macro("PairedRows", rows([
        (ref, *(entry["objets"].get(key, 0) for key in
                ("deux_corrects", "alexis_seul_correct", "robin_seul_correct", "deux_erreurs")))
        for ref, entry in paired.items()
    ])))
    texts.append(macro("ExtraRows", rows([
        (ref, *(entry["boites_en_trop"].get(key, 0) for key in
                ("alexis", "robin", "paires_memes_regions")))
        for ref, entry in paired.items()
    ])))
    (FIG/"tableaux.tex").write_text("".join(texts))


if __name__ == "__main__":
    main()
