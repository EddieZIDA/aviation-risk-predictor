"""
services/prediction_service.py — Service de Prédiction ML
==========================================================
Charge les artefacts produits par les notebooks et expose predict_accident().

Flux complet (identique au NB03 → NB05) :
  payload JSON (format CSV nettoyé / document MongoDB)
    → build_model_input()          # typage + feature engineering NB03 (FEATURE_COLS)
    → preprocessor.transform()     # ColumnTransformer NB03 (encodage one-hot, scaling)
    → variance_threshold.transform # sélection NB03 (feature_names.csv)
    → mapie.predict(alpha)         # classe majoritaire + ensemble de prédiction
    → dict résultat

Artefacts (dossier configurable via ARTIFACTS_DIR, défaut = notebooks/) :
  outputs/preprocessing_pipeline.pkl
  outputs/variance_threshold.pkl
  outputs/feature_names.csv
  uncertainty_outputs/mapie_classifier_lac.pkl   (sinon baseline_models/best_model.pkl)
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import settings
from utils.feature_engineering import FEATURE_COLS, build_model_input

logger = logging.getLogger(__name__)

# Ordre identique à INJURY_ORDER du NB03 : {NONE:0, MINR:1, SERS:2, FATL:3}
RISK_LABELS: dict[int, str] = {0: "NONE", 1: "MINR", 2: "SERS", 3: "FATL"}

# Niveau de risque MAPIE : alpha = 0.10 → couverture cible de 90 %
DEFAULT_ALPHA = 0.10

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "notebooks"


class PredictionService:
    """Pipeline d'inférence complet : feature engineering → preprocessing → MAPIE."""

    def __init__(self, artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
                 alpha: float = DEFAULT_ALPHA) -> None:
        self._dir = Path(artifacts_dir)
        self._alpha = alpha

        self._preprocessor = self._load(self._dir / "outputs" / "preprocessing_pipeline.pkl")
        self._vt = self._load(self._dir / "outputs" / "variance_threshold.pkl")
        self._feature_names = self._load_feature_names()

        mapie_path = self._dir / "uncertainty_outputs" / "mapie_classifier_lac.pkl"
        base_path = self._dir / "baseline_models" / "best_model.pkl"
        self._model = self._load(mapie_path if mapie_path.exists() else base_path)
        self.is_mapie = hasattr(self._model, "conformity_scores_") or "Mapie" in type(self._model).__name__

        self._check_consistency()

    # ── Chargement ───────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(
                f"Artefact introuvable : '{path}'. Exécutez les notebooks 03 à 05."
            )
        try:
            obj = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Échec du chargement de '{path}' : {exc}") from exc
        logger.info("Artefact chargé : %s", path)
        return obj

    def _load_feature_names(self) -> list[str]:
        path = self._dir / "outputs" / "feature_names.csv"
        if not path.exists():
            raise FileNotFoundError(f"Artefact introuvable : '{path}'.")
        return pd.read_csv(path).iloc[:, 0].tolist()

    def _check_consistency(self) -> None:
        """
        Vérifie au démarrage que les artefacts s'emboîtent. Un écart signifie
        que le code et les notebooks ont divergé : on refuse de démarrer plutôt
        que de servir des prédictions fausses.
        """
        expected_in = list(self._preprocessor.feature_names_in_)
        if expected_in != FEATURE_COLS:
            raise RuntimeError(
                "FEATURE_COLS (utils/feature_engineering.py) ne correspond pas aux "
                "colonnes du preprocessing_pipeline.pkl. Ré-exécutez le NB03."
            )
        selected = list(np.asarray(self._preprocessor.get_feature_names_out())[self._vt.get_support()])
        if selected != self._feature_names:
            raise RuntimeError(
                "variance_threshold.pkl et feature_names.csv ne proviennent pas "
                "de la même exécution du NB03."
            )

    # ── Propriétés publiques (utilisées par /api/health) ─────────────────────

    @property
    def model_name(self) -> str:
        estimator = getattr(self._model, "estimator", self._model)
        return type(estimator).__name__

    @property
    def model_type(self) -> str:
        return "MAPIE (LAC)" if self.is_mapie else "Modèle de base"

    @property
    def confidence_level(self) -> float | None:
        return round(1.0 - self._alpha, 4) if self.is_mapie else None

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def transform(self, payload: dict) -> pd.DataFrame:
        """Payload JSON → matrice d'une ligne, identique à une ligne de X_test.csv."""
        X = build_model_input(payload)
        X_processed = self._vt.transform(self._preprocessor.transform(X))
        return pd.DataFrame(X_processed, columns=self._feature_names)

    def predict_accident(self, payload: dict) -> dict:
        """
        Returns:
            {
              "prediction":         str        — classe la plus probable
              "confidence_level":   float|None — 1 - alpha (None sans MAPIE)
              "uncertainty_set":    list[str]  — classes plausibles MAPIE
              "probabilities":      dict       — probabilité par classe
              "raw_prediction_idx": int        — indice brut (debug)
            }

        Raises:
            ValueError  : payload vide ou valeur mal typée.
            RuntimeError: erreur interne du pipeline.
        """
        if not payload or not isinstance(payload, dict):
            raise ValueError("Le payload doit être un dictionnaire non vide.")

        X = self.transform(payload)  # ValueError remonte telle quelle (→ 400)

        try:
            if self.is_mapie:
                y_pred, y_sets = self._model.predict(X, alpha=self._alpha)
                plausible = [RISK_LABELS[i] for i, inside in enumerate(y_sets[0, :, 0]) if inside]
            else:
                y_pred = self._model.predict(X)
                plausible = [RISK_LABELS[int(y_pred[0])]]
            proba = self._predict_proba(X)
        except Exception as exc:
            raise RuntimeError(f"Erreur lors de la prédiction : {exc}") from exc

        raw_idx = int(np.ravel(y_pred)[0])
        prediction = RISK_LABELS[raw_idx]
        if not plausible:
            logger.warning("Ensemble MAPIE vide pour cette observation.")

        logger.info("Prédiction : %s | ensemble MAPIE : %s", prediction, plausible)
        return {
            "prediction": prediction,
            "confidence_level": self.confidence_level,
            "uncertainty_set": plausible,
            "probabilities": proba,
            "raw_prediction_idx": raw_idx,
        }

    def _predict_proba(self, X: pd.DataFrame) -> dict[str, float]:
        # MAPIE (cv="prefit") conserve le classifieur d'origine dans single_estimator_
        estimator = (getattr(self._model, "single_estimator_", None)
                     or getattr(self._model, "estimator", self._model))
        if not hasattr(estimator, "predict_proba"):
            return {}
        proba = estimator.predict_proba(X)[0]
        return {RISK_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)}


# ── Singleton ─────────────────────────────────────────────────────────────────
try:
    prediction_service: PredictionService | None = PredictionService(
        settings.artifacts_dir, alpha=settings.mapie_alpha)
except (FileNotFoundError, RuntimeError) as e:
    logger.error("IMPOSSIBLE DE CHARGER LE PIPELINE : %s", e)
    prediction_service = None
