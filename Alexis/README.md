# Alexis

YOLO26n, annotations par instance, 298 images. Le modèle livré se trouve dans
`runs/fruits_prolongation/weights/best.pt`.

Depuis la racine du dépôt, pour réentraîner dans une copie sans les deux
dossiers d'entraînement livrés :

```sh
python Alexis/train.py --weights Robin/yolo26n.pt
python Alexis/train.py --prolonger
python Alexis/selection.py
```

Pour reproduire la segmentation du cas des guimauves :

```sh
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt -o Alexis/sam2.1_t.pt
python Alexis/segmenter.py Alexis/dataset/images/val/image266.jpg --sortie segmentation.jpg
python Alexis/segmenter.py --evaluer
```

Les mesures du cas sont dans `cas_banane`.
