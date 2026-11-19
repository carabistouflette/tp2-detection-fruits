# Comparaison Alexis et Robin

- `resultats.json` : mAP sur les tests d'origine et les onze images communes.
- `confusions.json` : matrices de confusion à confiance 0,25 et IoU 0,50.
- `audit_par_image.csv` et `audit_par_boite.csv` : annotations de Robin.
- `erreurs_appariees.csv` : erreurs des deux YOLO26n sur les mêmes objets.

Depuis la racine du dépôt :

```sh
python Alexis/comparer_robin.py
python Alexis/auditer_annotations.py
python Alexis/comparer_confusions.py
python Alexis/comparer_erreurs_communes.py
python Alexis/illustrer_erreurs_communes.py
python Alexis/preparer_figures_comparatives.py
```
