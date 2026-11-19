"""Chemins et constantes du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

APPLICATION_FILE = DATA_RAW / "application_train.csv"

TARGET = "TARGET"      # 1 = défaut de paiement, 0 = remboursé
ID_COL = "SK_ID_CURR"
RANDOM_STATE = 42
TEST_SIZE = 0.2
