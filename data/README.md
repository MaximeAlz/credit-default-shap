# Données

Les données ne sont pas versionnées : elles sont trop volumineuses et soumises aux règles de Kaggle.

1. Créer un compte sur [kaggle.com](https://www.kaggle.com).
2. Ouvrir la compétition [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) et accepter ses règles (onglet *Data*).
3. Télécharger l'archive (environ 700 Mo) et extraire les CSV dans `data/raw/`.

Avec l'API Kaggle :

```bash
pip install kaggle
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw
```

Fichier utilisé pour l'instant : `application_train.csv`.
Fichiers prévus ensuite : `bureau.csv`, `previous_application.csv`.
