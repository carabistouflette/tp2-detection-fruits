# Importer les bibliothèques nécessaires
from ultralytics import YOLO
from pathlib import Path

# Définir le dossier des résultats d'évaluation
output_dir = Path("evaluation_results").resolve()

# Charger les meilleurs poids du modèle entraîné
model = YOLO("runs/fruits_detection/weights/best.pt")

# Évaluer le modèle sur l'ensemble de test (non utilisé pendant l'entraînement)
metrics = model.val(
    data="dataset/data.yaml",
    split="test",
    imgsz=640,
    plots=True,
    project=str(output_dir),
    name="YOLO_Test",
    exist_ok=True,
)
print(f"Precision {metrics.box.mp:.3f} | Recall {metrics.box.mr:.3f} | "
      f"mAP50 {metrics.box.map50:.3f} | mAP50-95 {metrics.box.map:.3f}")
