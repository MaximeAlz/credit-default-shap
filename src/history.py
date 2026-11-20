"""Historique de crédit de chaque client, agrégé à une ligne par SK_ID_CURR.

- bureau.csv : crédits détenus auprès d'autres établissements (données de bureau de crédit).
- previous_application.csv : demandes passées chez Home Credit, accordées ou refusées.

Le nombre de variables est volontairement limité (une vingtaine, toutes lisibles)
pour que les explications SHAP restent interprétables par un analyste crédit.
"""
import numpy as np
import pandas as pd

from src.config import DATA_RAW, ID_COL


def _safe_ratio(num, den):
    return num / den.replace(0, np.nan)


def bureau_features(path=DATA_RAW / "bureau.csv") -> pd.DataFrame:
    b = pd.read_csv(path)
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["IS_OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)
    b["DEBT_ACTIVE"] = b["AMT_CREDIT_SUM_DEBT"].where(b["IS_ACTIVE"] == 1)
    b["CREDIT_ACTIVE_AMT"] = b["AMT_CREDIT_SUM"].where(b["IS_ACTIVE"] == 1)

    g = b.groupby(ID_COL)
    feats = pd.DataFrame({
        "BUR_NB_CREDITS": g.size(),                                # nb de crédits connus
        "BUR_NB_ACTIVE": g["IS_ACTIVE"].sum(),                     # crédits en cours
        "BUR_NB_OVERDUE": g["IS_OVERDUE"].sum(),                   # crédits en retard aujourd'hui
        "BUR_MAX_DAYS_OVERDUE": g["CREDIT_DAY_OVERDUE"].max(),
        "BUR_MAX_AMT_OVERDUE": g["AMT_CREDIT_MAX_OVERDUE"].max(),  # pire retard historique (montant)
        "BUR_TOTAL_DEBT": g["DEBT_ACTIVE"].sum(),                  # dette restante sur crédits actifs
        "BUR_TOTAL_CREDIT_ACTIVE": g["CREDIT_ACTIVE_AMT"].sum(),
        "BUR_DAYS_LAST_CREDIT": g["DAYS_CREDIT"].max(),            # ancienneté du crédit le plus récent
        "BUR_DAYS_FIRST_CREDIT": g["DAYS_CREDIT"].min(),           # profondeur de l'historique
        "BUR_NB_PROLONG": g["CNT_CREDIT_PROLONG"].sum(),           # prolongations demandées
    })
    # Part de la dette encore due sur les crédits actifs
    feats["BUR_DEBT_RATIO"] = _safe_ratio(feats["BUR_TOTAL_DEBT"], feats["BUR_TOTAL_CREDIT_ACTIVE"])
    return feats.drop(columns=["BUR_TOTAL_CREDIT_ACTIVE"])


def previous_features(path=DATA_RAW / "previous_application.csv") -> pd.DataFrame:
    usecols = [ID_COL, "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT",
               "DAYS_DECISION", "CNT_PAYMENT", "RATE_DOWN_PAYMENT"]
    p = pd.read_csv(path, usecols=usecols)
    p["IS_APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["IS_REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    # Montant accordé / montant demandé : < 1 si la banque a réduit la demande
    p["GRANTED_ASKED_RATIO"] = _safe_ratio(p["AMT_CREDIT"], p["AMT_APPLICATION"])
    p["DAYS_DECISION_REFUSED"] = p["DAYS_DECISION"].where(p["IS_REFUSED"] == 1)

    g = p.groupby(ID_COL)
    feats = pd.DataFrame({
        "PREV_NB_APPLICATIONS": g.size(),
        "PREV_NB_REFUSED": g["IS_REFUSED"].sum(),
        "PREV_APPROVAL_RATE": g["IS_APPROVED"].mean(),
        "PREV_DAYS_LAST_DECISION": g["DAYS_DECISION"].max(),
        "PREV_DAYS_LAST_REFUSAL": g["DAYS_DECISION_REFUSED"].max(),  # refus récent = signal fort
        "PREV_MEAN_GRANTED_ASKED": g["GRANTED_ASKED_RATIO"].mean(),
        "PREV_MEAN_CNT_PAYMENT": g["CNT_PAYMENT"].mean(),            # durée moyenne demandée
        "PREV_MEAN_DOWN_PAYMENT_RATE": g["RATE_DOWN_PAYMENT"].mean(),
    })
    return feats


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Joint les agrégats d'historique. Un client absent d'une table n'a simplement
    pas d'historique : ses compteurs valent 0, les autres agrégats restent NaN."""
    df = df.merge(bureau_features(), left_on=ID_COL, right_index=True, how="left")
    df = df.merge(previous_features(), left_on=ID_COL, right_index=True, how="left")
    count_cols = ["BUR_NB_CREDITS", "BUR_NB_ACTIVE", "BUR_NB_OVERDUE", "BUR_NB_PROLONG",
                  "PREV_NB_APPLICATIONS", "PREV_NB_REFUSED"]
    df[count_cols] = df[count_cols].fillna(0)

    # Endettement externe rapporté au revenu déclaré
    df["BUR_DEBT_INCOME_RATIO"] = _safe_ratio(df["BUR_TOTAL_DEBT"], df["AMT_INCOME_TOTAL"])
    return df
