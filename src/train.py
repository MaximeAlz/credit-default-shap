"""Entraînement : régression logistique (référence) puis LightGBM.

Usage :
    python -m src.train
    python -m src.train --data chemin/vers/application_train.csv --sample 50000
    python -m src.train --no-history   # sans les tables d'historique
"""
import argparse
import json

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import APPLICATION_FILE, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from src.data import load_application
from src.evaluate import credit_metrics
from src.features import add_domain_features, split_xy
from src.history import add_history


def build_logistic(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="category").columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]), cat_cols),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
    return Pipeline([("pre", pre), ("clf", clf)])


def build_lightgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def main(data_path=APPLICATION_FILE, sample=None, history=True):
    df = add_domain_features(load_application(data_path))
    if history:
        df = add_history(df)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
    X, y = split_xy(df)
    print(f"{len(X):,} clients, {X.shape[1]} variables, taux de défaut = {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Jeu de validation interne pour l'early stopping (le test reste intouché)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    results = {}

    print("→ Régression logistique (référence)")
    logit = build_logistic(X_train).fit(X_train, y_train)
    results["logistic_regression"] = credit_metrics(y_test, logit.predict_proba(X_test)[:, 1])

    print("→ LightGBM")
    gbm = build_lightgbm()
    gbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    results["lightgbm"] = credit_metrics(y_test, gbm.predict_proba(X_test)[:, 1])
    results["lightgbm"]["best_iteration"] = int(gbm.best_iteration_)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(logit, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(gbm, MODELS_DIR / "lightgbm.joblib")
    joblib.dump((X_test, y_test), MODELS_DIR / "test_set.joblib")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    print(pd.DataFrame(results).T.to_string())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=APPLICATION_FILE)
    parser.add_argument("--sample", type=int, default=None,
                        help="Sous-échantillon pour itérer vite (ex. 50000)")
    parser.add_argument("--no-history", action="store_true",
                        help="N'utiliser que application_train (sans bureau / previous_application)")
    args = parser.parse_args()
    main(args.data, args.sample, history=not args.no_history)
