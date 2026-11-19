# TP2 · Détection de fruits

Alexis ROBIN et Robin FOEX, M1 MIAS.

| Partie | Contenu |
|---|---|
| [Alexis](Alexis/) | YOLO26n, 20 puis 10 époques. Boîtes par fruit visible, essais sur photos externes et segmentation avec SAM. |
| [Robin](Robin/) | YOLO26n et YOLO11n sur son jeu d'annotations. Notebook, poids, courbes et prédictions. |
| [Comparaison](comparaison/) | Mesures, matrices de confusion, audit des annotations et erreurs sur les images communes. |

Les deux dossiers contiennent `dataset/data.yaml`, les images et annotations,
`train.py`, `evaluate.py`, `predict.py` et les fichiers `best.pt` sous `runs/`.
Les courbes d'entraînement sont sous `runs/`, les graphiques de test sous
`evaluation_results/` et `comparaison/`, et les images prédites sous
`predictions/`.

## Relancer les prédictions

Depuis la racine, dans un environnement Python avec PyTorch installé :

```sh
python -m pip install -r requirements.txt
python Alexis/evaluate.py
python Alexis/predict.py
(cd Robin && python evaluate.py && python predict.py)
python Alexis/comparer_robin.py
```

Pour refaire les entraînements, voir [Alexis](Alexis/README.md) et
[Robin](Robin/README.md). Le compte rendu est remis séparément.
Le ZIP demandé dans le sujet est disponible dans les
[versions du dépôt](https://github.com/carabistouflette/tp2-detection-fruits/releases).
