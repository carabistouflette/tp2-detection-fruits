# Importer YOLO depuis la bibliothèque Ultralytics
import time
from pathlib import Path
from ultralytics import YOLO

# Charger un modèle YOLO26n pré-entraîné (transfer learning)
model = YOLO("yolo26n.pt")

# Suivi : une ligne par époque avec son temps et les métriques de validation
t0 = t_prev = time.time()
start_epoch = 0
def log_epoch(trainer):
    global t_prev
    now = time.time()
    m = trainer.metrics
    done = trainer.epoch + 1 - start_epoch
    eta = (now - t0) / done * (trainer.epochs - trainer.epoch - 1) / 60   # shortcut: majorant, l'early stopping peut couper avant
    print(f"EPOCH {trainer.epoch + 1}/{trainer.epochs} | {now - t_prev:.0f}s (total {(now - t0) / 60:.1f} min, reste ~{eta:.1f} min max) | "
          f"mAP50 {m['metrics/mAP50(B)']:.3f} | mAP50-95 {m['metrics/mAP50-95(B)']:.3f} | "
          f"meilleur fitness {trainer.best_fitness:.3f}", flush=True)
    t_prev = now
model.add_callback("on_fit_epoch_end", log_epoch)

# Lancer le fine-tuning sur notre dataset
model.train(
    data="dataset/data.yaml",  # Configuration du dataset
    epochs=30,                 # Maximum d'époques (le sujet propose 20) : l'early stopping peut couper avant
    patience=5,                # Early stopping : arrêt si la métrique de validation ne progresse plus pendant 5 époques
    imgsz=640,                 # Taille d'entrée des images
    batch=8,                   # Nombre d'images par lot
    seed=42,                   # Reproductibilité
    project=str(Path("runs").resolve()),  # Chemin absolu : sinon Ultralytics range tout dans runs/detect/runs/
    name="fruits_detection",   # Nom de l'expérience
    exist_ok=True,
)
