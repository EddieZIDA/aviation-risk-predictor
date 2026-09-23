"""
Configuration pytest commune.

Lancer depuis la racine du projet ou depuis backend/ :
    python -m pytest backend/tests
"""

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Les modules du backend s'importent comme « services.x », « utils.x »...
sys.path.insert(0, str(BACKEND_DIR))

CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "ntsb_clean_final.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "notebooks"

requires_artifacts = pytest.mark.skipif(
    not (CLEAN_CSV.exists() and (ARTIFACTS_DIR / "outputs" / "X_test.csv").exists()),
    reason="Données ou artefacts des notebooks absents (exécuter NB01 → NB05).",
)


@pytest.fixture(scope="session")
def clean_df():
    import pandas as pd
    return pd.read_csv(CLEAN_CSV, low_memory=False)


@pytest.fixture(scope="session")
def service():
    from services.prediction_service import PredictionService
    return PredictionService(ARTIFACTS_DIR)
