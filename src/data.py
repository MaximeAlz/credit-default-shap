"""Chargement et nettoyage minimal des données Home Credit."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import APPLICATION_FILE

# Valeur sentinelle connue du jeu Home Credit : 365243 jours (~1000 ans)
# dans DAYS_EMPLOYED signifie « sans emploi / retraité », pas une vraie durée.
DAYS_EMPLOYED_ANOMALY = 365243


def load_application(path: Path = APPLICATION_FILE) -> pd.DataFrame:
    """Charge application_train.csv et corrige les anomalies connues."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} introuvable. Voir data/README.md pour télécharger les données."
        )
    df = pd.read_csv(path)

    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)

    # Genre « XNA » : 4 lignes seulement, on les retire
    if "CODE_GENDER" in df.columns:
        df = df[df["CODE_GENDER"] != "XNA"]

    return df.reset_index(drop=True)
