"""Comparaison SHAP / LIME sur le modèle LightGBM.

Trois questions :
1. Accord : les deux méthodes désignent-elles les mêmes variables comme
   les plus importantes pour un client donné ? (recouvrement du top 5, signe,
   corrélation de rang de Spearman sur toutes les variables)
2. Stabilité : si on relance l'explication du même client, obtient-on la même
   réponse ? SHAP (TreeExplainer) est exact donc déterministe ; LIME repose sur
   un échantillonnage aléatoire autour du client.
3. Coût : temps de calcul par explication.

Pour comparer les deux méthodes sur exactement les mêmes points, on retient des
clients sans valeur manquante dans les variables les plus importantes, puis on
impute les rares valeurs manquantes restantes (médiane / modalité la plus
fréquente) : LIME ne sait pas perturber une valeur manquante.

Usage (après src.train) :
    python -m src.compare_explainers
"""
import json
import time
from itertools import combinations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR

N_CLIENTS = 100        # clients pour la mesure d'accord
N_STABILITY = 20       # clients pour la mesure de stabilité
N_RUNS = 5             # relances de LIME par client
TOP_K = 5
LIME_SAMPLES = 5000    # valeur par défaut de LIME

COLOR_SHAP = "#2a78d6"
COLOR_LIME = "#eb6834"
INK = "#3d3d3a"
MUTED = "#8a8980"


# ---------------------------------------------------------------------------
# Préparation : LIME travaille sur des tableaux numériques
# ---------------------------------------------------------------------------
class Encoded:
    """Passe d'un DataFrame (catégories pandas) à une matrice numérique et retour."""

    def __init__(self, X_ref: pd.DataFrame):
        self.columns = X_ref.columns
        self.cat_cols = X_ref.select_dtypes(include="category").columns.tolist()
        self.cat_idx = [self.columns.get_loc(c) for c in self.cat_cols]
        self.categories = {c: X_ref[c].cat.categories for c in self.cat_cols}
        self.medians = X_ref.select_dtypes(include="number").median()
        self.modes = {c: X_ref[c].mode().iloc[0] for c in self.cat_cols}

    def impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        num = X.select_dtypes(include="number").columns
        X[num] = X[num].fillna(self.medians[num])
        for c in self.cat_cols:
            X[c] = X[c].fillna(self.modes[c])
        return X

    def to_array(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].cat.codes
        return X.to_numpy(dtype=float)

    def to_frame(self, A: np.ndarray) -> pd.DataFrame:
        X = pd.DataFrame(A, columns=self.columns)
        for c in self.cat_cols:
            codes = np.clip(np.rint(X[c]).astype(int), 0, len(self.categories[c]) - 1)
            X[c] = pd.Categorical.from_codes(codes, categories=self.categories[c])
        return X


def top_k(weights: np.ndarray, k: int = TOP_K) -> set:
    return set(np.argsort(-np.abs(weights))[:k])


def lime_weights(explainer, row, predict_fn, n_features, seed) -> np.ndarray:
    explainer.random_state = np.random.RandomState(seed)
    exp = explainer.explain_instance(row, predict_fn, num_features=n_features,
                                     num_samples=LIME_SAMPLES, labels=(1,))
    w = np.zeros(n_features)
    for idx, val in exp.as_map()[1]:
        w[idx] = val
    return w


def main():
    gbm = joblib.load(MODELS_DIR / "lightgbm.joblib")
    X_test, _ = joblib.load(MODELS_DIR / "test_set.joblib")
    names = X_test.columns.tolist()
    n_feat = len(names)

    # Clients complets sur les 10 variables les plus importantes (selon le gain LightGBM)
    top10 = pd.Series(gbm.booster_.feature_importance("gain"), index=names).nlargest(10).index
    pool = X_test.dropna(subset=top10)
    clients = pool.sample(n=N_CLIENTS, random_state=RANDOM_STATE)

    enc = Encoded(X_test)
    background = enc.to_array(enc.impute(X_test.sample(n=20000, random_state=RANDOM_STATE)))
    clients_imp = enc.impute(clients)
    clients_arr = enc.to_array(clients_imp)

    def predict_fn(A):
        return gbm.predict_proba(enc.to_frame(A))

    lime_exp = LimeTabularExplainer(
        background, feature_names=names, categorical_features=enc.cat_idx,
        class_names=["rembourse", "defaut"], mode="classification",
        discretize_continuous=True, random_state=RANDOM_STATE,
    )

    # --- SHAP ---
    t0 = time.perf_counter()
    shap_vals = shap.TreeExplainer(gbm)(clients_imp).values
    t_shap = (time.perf_counter() - t0) / N_CLIENTS

    # --- LIME ---
    t0 = time.perf_counter()
    lime_vals = np.array([lime_weights(lime_exp, r, predict_fn, n_feat, RANDOM_STATE)
                          for r in clients_arr])
    t_lime = (time.perf_counter() - t0) / N_CLIENTS

    # --- 1. Accord ---
    overlap, sign_agree, rho = [], [], []
    for s, l in zip(shap_vals, lime_vals):
        ts, tl = top_k(s), top_k(l)
        common = ts & tl
        overlap.append(len(common) / TOP_K)
        if common:
            sign_agree.append(np.mean([np.sign(s[j]) == np.sign(l[j]) for j in common]))
        rho.append(spearmanr(np.abs(s), np.abs(l)).statistic)

    # --- 2. Stabilité de LIME (SHAP est déterministe : stabilité = 1) ---
    stability = []
    for r in clients_arr[:N_STABILITY]:
        runs = [top_k(lime_weights(lime_exp, r, predict_fn, n_feat, seed))
                for seed in range(N_RUNS)]
        stability.append(np.mean([len(a & b) / TOP_K for a, b in combinations(runs, 2)]))
    shap_check = shap.TreeExplainer(gbm)(clients_imp.iloc[:N_STABILITY]).values
    shap_deterministic = bool(np.allclose(shap_check, shap_vals[:N_STABILITY]))

    results = {
        "n_clients": N_CLIENTS,
        "top_k": TOP_K,
        "accord_top5_moyen": round(float(np.mean(overlap)), 3),
        "accord_signe_variables_communes": round(float(np.mean(sign_agree)), 3),
        "spearman_moyen_toutes_variables": round(float(np.nanmean(rho)), 3),
        "stabilite_top5_lime": round(float(np.mean(stability)), 3),
        "stabilite_top5_shap": 1.0 if shap_deterministic else None,
        "temps_par_explication_s": {"shap": round(t_shap, 4), "lime": round(t_lime, 3)},
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "shap_vs_lime.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # --- Figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_summary(overlap, stability)
    _plot_one_client(shap_vals[0], lime_vals[0], names)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def _plot_summary(overlap, stability):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.arange(0, 1.2, 0.2)
    labels = [f"{int(b * TOP_K)}/{TOP_K}" for b in bins]

    ax = axes[0]
    counts = [np.isclose(overlap, b).sum() for b in bins]
    ax.bar(labels, counts, color=COLOR_SHAP, width=0.6)
    ax.set_title("Accord SHAP / LIME sur le top 5\n(variables en commun, par client)",
                 fontsize=10, color=INK, loc="left")
    ax.set_ylabel("Nombre de clients", color=INK, fontsize=9)
    ax.yaxis.grid(True, color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    ax = axes[1]
    vals = [1.0, float(np.mean(stability))]
    bars = ax.barh(["SHAP", "LIME"], vals, color=[COLOR_SHAP, COLOR_LIME], height=0.5)
    for b, v in zip(bars, vals):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.set_title(f"Stabilité : part du top 5 conservée d'une relance à l'autre\n"
                 f"({N_RUNS} relances × {len(stability)} clients)",
                 fontsize=10, color=INK, loc="left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_vs_lime_summary.png", dpi=150)
    plt.close(fig)


def _plot_one_client(s, l, names, n=8):
    """Top variables d'un client selon chaque méthode, normalisées pour être comparables."""
    idx = sorted(top_k(s, n) | top_k(l, n), key=lambda j: -abs(s[j]))[:10]
    s_n = s[idx] / np.abs(s).max()
    l_n = l[idx] / np.abs(l).max()
    y = np.arange(len(idx))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y - 0.2, s_n, height=0.36, color=COLOR_SHAP, label="SHAP")
    ax.barh(y + 0.2, l_n, height=0.36, color=COLOR_LIME, label="LIME")
    ax.set_yticks(y, [names[j] for j in idx])
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Contribution au risque (normalisée par méthode)\n"
                  "← diminue le risque   |   augmente le risque →", color=INK, fontsize=9)
    ax.set_title("Un même client expliqué par SHAP et par LIME",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_vs_lime_client.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
