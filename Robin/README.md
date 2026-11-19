# Robin

YOLO26n et YOLO11n, 300 images, avec des boîtes groupées dans certaines scènes.
Le notebook `TP2_yolo.ipynb` est prévu pour être ouvert depuis ce dossier.

- YOLO26n : `runs/fruits_detection/weights/best.pt`.
- YOLO11n : `runs/fruits_yolo11n/weights/best.pt`.

Pour réentraîner dans une copie sans les dossiers de runs livrés :

```sh
cd Robin
python train.py
python train_yolo11.py
```

Les mesures comparatives et les graphiques de test de YOLO11n sont dans
le dossier `comparaison` à la racine.
