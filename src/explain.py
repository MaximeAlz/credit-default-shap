"""Explicabilité SHAP du modèle LightGBM.

- Global : quelles variables pèsent le plus sur le risque, et dans quel sens.
- Local : pour un client donné, les raisons principales de son score
  (équivalent des « reason codes » exigés quand une banque refuse un crédit).

Usage (après src.train) :
    python -m src.explain
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE


def load_artifacts():
    gbm = joblib.load(MODELS_DIR / "lightgbm.joblib")
    X_test, y_test = joblib.load(MODELS_DIR / "test_set.joblib")
    return gbm, X_test, y_test


def compute_shap(model, X: pd.DataFrame) -> shap.Explanation:
    """Valeurs SHAP exactes pour un modèle d'arbres (TreeExplainer), en log-odds."""
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def reason_codes(explanation: shap.Explanation, i: int, top: int = 3) -> pd.DataFrame:
    """Les `top` variables qui augmentent le plus le risque du client i."""
    contrib = pd.DataFrame({
        "variable": explanation.feature_names,
        "valeur": explanation.data[i],
        "contribution_shap": explanation.values[i],
    })
    return (contrib[contrib["contribution_shap"] > 0]
            .sort_values("contribution_shap", ascending=False)
            .head(top)
            .reset_index(drop=True))


def main(n_sample: int = 5000):
    gbm, X_test, y_test = load_artifacts()
    X_s = X_test.sample(n=min(n_sample, len(X_test)), random_state=RANDOM_STATE)
    expl = compute_shap(gbm, X_s)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(expl, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=150)
    plt.close()

    shap.plots.bar(expl, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_importance.png", dpi=150)
    plt.close()

    # Exemple local : le client jugé le plus risqué de l'échantillon
    proba = gbm.predict_proba(X_s)[:, 1]
    i = int(proba.argmax())
    shap.plots.waterfall(expl[i], max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall_client.png", dpi=150)
    plt.close()

    print(f"Client le plus risqué : probabilité de défaut = {proba[i]:.1%}")
    print(reason_codes(expl, i).to_string())
    print(f"Figures enregistrées dans {FIGURES_DIR}")


if __name__ == "__main__":
    main()
