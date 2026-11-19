"""Variables métier et préparation de la matrice X."""
import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios classiques en scoring crédit (capacité de remboursement, endettement)."""
    df = df.copy()
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    credit = df["AMT_CREDIT"].replace(0, np.nan)

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income       # montant emprunté / revenu
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income     # taux d'effort
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / credit              # ~ 1 / durée du prêt
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / credit   # apport
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    return df


def split_xy(df: pd.DataFrame):
    """Sépare X / y et passe les colonnes texte en type 'category' (géré nativement par LightGBM)."""
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
    return X, y
