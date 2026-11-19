# Importer les bibliothèques nécessaires
from ultralytics import YOLO
from pathlib import Path

# Charger le modèle entraîné
model = YOLO("runs/fruits_detection/weights/best.pt")

# Définir le dossier des prédictions
output_dir = Path("predictions").resolve()

# Détecter les fruits dans les images de test
results = model.predict(
    source="dataset/images/test",
    conf=0.25,
    save=True,
    project=str(output_dir),
    name="YOLO_Predictions",
    exist_ok=True,
)
