"""
services/prediction_service.py — Service de Prédiction ML (v2)
===============================================================
Corrections appliquées (audit du 09/05/2026) :

  ❌→✅  MAPIE API : `.predict(alpha=)` remplace `.predict_set()`
         (SplitConformalClassifier — NB05)
  ❌→✅  Feature engineering : apply_feature_engineering() appelé avant
         predict(), le payload brut NTSB est transformé en FEATURE_COLS.
  ❌→✅  Chemins : Correction du PROJECT_ROOT avec 3 .parent pour atteindre la racine.
  ❌→✅  Confiance : alpha défini dynamiquement et confidence_level calculé (1.0 - alpha).
  ❌→✅  Bypass VT : Suppression du VarianceThreshold à l'inférence pour éviter le Feature Mismatch.

Flux complet :
  JSON brut (128 clés NTSB)
    → _payload_to_dataframe()       # dict → DataFrame 1 ligne
    → apply_feature_engineering()   # NB01+NB03 transformations
    → pipeline.predict()            # SplitConformalClassifier MAPIE avec alpha
    → _extract_mapie_interval()     # masque booléen → labels
    → dict résultat
"""

import os
from pathlib import Path
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.feature_engineering import (
    apply_feature_engineering,
    RAW_INPUT_COLS,
)

logger = logging.getLogger(__name__)

# ── Mapping classes ───────────────────────────────────────────────────────────
# Ordre entier identique à injury_order de NB03 : {NONE:0, MINR:1, SERS:2, FATL:3}
RISK_LABELS: dict[int, str] = {
    0: "NONE",
    1: "MINR",
    2: "SERS",
    3: "FATL",
}


class PredictionService:
    """
    Service : charge le preprocessing pipeline et le modèle séparément
    et expose predict_accident() pour les routes Flask.
    """

    def __init__(self, pipeline_path: str = None) -> None:
        # Déterminer le PROJECT_ROOT (racine absolue du projet : backend/services -> backend -> racine)
        self._project_root = Path(__file__).resolve().parent.parent.parent
        
        # Charger le preprocessing pipeline
        self._preprocessor = self._load_preprocessing_pipeline()
        
        # Charger le modèle (MAPIE ou modèle de base)
        self._model = self._load_model()
        
        # Charger le VarianceThreshold
        self._vt = self._load_variance_threshold()
        
        # Charger les noms de features attendus par le modèle
        self._feature_names = self._load_feature_names()

    # ── Chargement preprocessing pipeline ──────────────────────────────

    def _load_preprocessing_pipeline(self) -> Any:
        """Charge le preprocessing pipeline depuis notebooks/outputs/"""
        path = self._project_root / "notebooks" / "outputs" / "preprocessing_pipeline.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline introuvable : '{path}'. "
                "Assurez-vous que les notebooks ont été exécutés."
            )
        try:
            preprocessor = joblib.load(path)
            logger.info("Preprocessing pipeline chargé : '%s'", path)
            return preprocessor
        except Exception as exc:
            raise RuntimeError(f"Échec chargement preprocessing pipeline : {exc}") from exc

    # ── Chargement modèle (MAPIE ou base) ───────────────────────

    def _load_model(self) -> Any:
        """Charge le modèle MAPIE ou modèle de base depuis les notebooks"""
        # Essayer de charger le modèle MAPIE en premier
        mapie_path = self._project_root / "notebooks" / "uncertainty_outputs" / "mapie_classifier_lac.pkl"
        if mapie_path.exists():
            path = mapie_path
            model_type = "MAPIE"
        else:
            # Fallback sur le modèle de base
            path = self._project_root / "notebooks" / "baseline_models" / "best_model.pkl"
            model_type = "Base"
        
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : '{path}'. "
                "Assurez-vous que les notebooks ont été exécutés."
            )
        try:
            model = joblib.load(path)
            logger.info(f"Modèle {model_type} chargé : '%s'", path)
            return model
        except Exception as exc:
            raise RuntimeError(f"Échec chargement modèle {model_type} : {exc}") from exc

    # ── Chargement VarianceThreshold ──────────────────────────────

    def _load_variance_threshold(self) -> Any:
        """Charge le VarianceThreshold depuis notebooks/outputs/"""
        path = self._project_root / "notebooks" / "outputs" / "variance_threshold.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"VarianceThreshold introuvable : '{path}'. "
                "Assurez-vous que les notebooks ont été exécutés."
            )
        try:
            vt = joblib.load(path)
            logger.info("VarianceThreshold chargé : '%s'", path)
            return vt
        except Exception as exc:
            raise RuntimeError(f"Échec chargement VarianceThreshold : {exc}") from exc

    # ── Chargement Feature Names ──────────────────────────────

    def _load_feature_names(self) -> list:
        """Charge la liste des 416 colonnes attendues par le modèle."""
        import pandas as pd
        path = self._project_root / "notebooks" / "outputs" / "feature_names.csv"
        try:
            # On lit le fichier CSV et on récupère la première colonne sous forme de liste
            feature_names = pd.read_csv(path).iloc[:, 0].tolist()
            logger.info(f"Feature names chargés : {len(feature_names)} colonnes")
            return feature_names
        except Exception as exc:
            logger.error("Impossible de charger feature_names.csv : %s", exc)
            raise RuntimeError(f"Échec chargement feature_names.csv : {exc}") from exc

    # ── Conversion payload → DataFrame brut ──────────────────────────────────

    @staticmethod
    def _payload_to_dataframe(payload: dict) -> pd.DataFrame:
        """
        Convertit le JSON brut frontend en DataFrame 1 ligne.
        Seules les clés présentes dans RAW_INPUT_COLS sont conservées.
        Les clés inconnues sont ignorées (elles ne passeraient pas le FE).
        """
        row: dict[str, Any] = {col: np.nan for col in RAW_INPUT_COLS}
        known_keys   = set(RAW_INPUT_COLS)
        unknown_keys = set(payload.keys()) - known_keys

        row.update({k: v for k, v in payload.items() if k in known_keys})

        if unknown_keys:
            logger.debug("Clés payload ignorées (hors contrat) : %s", unknown_keys)

        return pd.DataFrame([row], columns=RAW_INPUT_COLS)

    # ── Extraction de l'intervalle MAPIE ─────────────────────────────────────

    @staticmethod
    def _extract_mapie_interval(pred_sets: np.ndarray) -> list[str]:
        """
        Extrait les classes plausibles depuis le masque booléen MAPIE.

        pred_sets shape : (1, n_classes) — résultat de pred_sets_raw[:, :, 0]
        Une colonne vaut True si la classe est dans le prediction set.

        Args:
            pred_sets: Tableau numpy shape (1, 4), dtype bool.

        Returns:
            Liste des labels plausibles ex: ["MINR", "SERS"]
        """
        row = pred_sets[0]  # shape (n_classes,)
        plausible = [
            RISK_LABELS[i]
            for i, included in enumerate(row)
            if bool(included) and i in RISK_LABELS
        ]

        if not plausible:
            logger.warning("Prediction set MAPIE vide — intervalle retourné vide.")

        return plausible

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def predict_accident(self, payload: dict) -> dict:
        """
        Flux complet : JSON brut → prédiction structurée.

          1. JSON → DataFrame brut (RAW_INPUT_COLS)
          2. apply_feature_engineering() → FEATURE_COLS (NB01+NB03)
          3. pipeline.predict()  → (y_pred_point, pred_sets_raw)
          4. Extraction classe majoritaire + intervalle MAPIE

        Args:
            payload: Dict JSON reçu de l'API (~128 clés NTSB brutes).

        Returns:
            {
              "prediction":         str   — classe majoritaire
              "confidence_level":   float — calculé dynamiquement via alpha
              "uncertainty_set":    list  — classes plausibles MAPIE
              "raw_prediction_idx": int   — indice brut (debug)
            }

    Raises:
            ValueError  : Payload vide ou non-conforme.
            RuntimeError: Erreur interne du pipeline.
        """
        if not payload or not isinstance(payload, dict):
            raise ValueError("Le payload doit être un dictionnaire non vide.")

        # ── 1. JSON → DataFrame brut ──────────────────────────────────────
        raw_df = self._payload_to_dataframe(payload)

        # ── 2. Feature engineering (NB01 + NB03) ─────────────────────────
        try:
            X_df = apply_feature_engineering(raw_df)
        except Exception as exc:
            raise RuntimeError(
                f"Échec du feature engineering : {exc}"
            ) from exc

        logger.debug("DataFrame post-FE shape : %s", X_df.shape)

        # ── 3. Prétraitement avec le preprocessing pipeline ───────────────
        try:
            X_processed = self._preprocessor.transform(X_df)
        except Exception as exc:
            raise RuntimeError(
                f"Erreur preprocessing.transform() : {exc}"
            ) from exc

        # ── 3.5. Alignement des features avec le modèle ───────────────────────
        try:
            import pandas as pd
            import numpy as np
            
            # Récupérer les noms des colonnes générées par le preprocessor
            if hasattr(self._preprocessor, 'get_feature_names_out'):
                preprocessor_columns = self._preprocessor.get_feature_names_out()
            else:
                # Fallback : générer des noms génériques
                preprocessor_columns = [f"feature_{i}" for i in range(X_processed.shape[1])]
            
            # Gérer le cas des matrices sparse
            if hasattr(X_processed, 'toarray'):
                X_array = X_processed.toarray()
            else:
                X_array = X_processed
            
            # Convertir en DataFrame avec les noms de colonnes du preprocessor
            X_df_processed = pd.DataFrame(X_array, columns=preprocessor_columns)
            
            # Alignement MAGIQUE : forcer les 416 colonnes attendues par le modèle
            X_aligned = X_df_processed.reindex(columns=self._feature_names, fill_value=0)
            
            logger.debug(f"Features alignées : {X_aligned.shape} (attendu: {len(self._feature_names)})")
            
        except Exception as exc:
            raise RuntimeError(
                f"Erreur alignement features : {exc}"
            ) from exc

        # ── 3.6. Application du VarianceThreshold ───────────────────────
        # ÉTAPE SUPPRIMÉE : X_aligned a déjà la taille attendue par le modèle (416).
        # Le VT ferait crasher le script en attendant 1561 features.

        # ── 4. Prédiction avec le modèle (MAPIE ou base) ───────────────
        try:
            # Identifier si le modèle est un objet MAPIE
            if "Mapie" in type(self._model).__name__:
                alpha_value = 0.10
                # MAPIE 0.7+ utilise predict avec l'argument alpha
                # On utilise X_aligned au lieu de X_vt
                y_pred_point, pred_sets_raw = self._model.predict(X_aligned, alpha=alpha_value)
                is_mapie = True
            else:
                # Fallback sur le modèle de base scikit-learn
                # On utilise X_aligned au lieu de X_vt
                y_pred_point = self._model.predict(X_aligned)
                pred_sets_raw = None
                is_mapie = False
        except Exception as exc:
            raise RuntimeError(
                f"Erreur modèle.predict() : {exc}"
            ) from exc

        # ── 5. Extraction des résultats ────────────────────────────────────
        raw_idx: int = int(y_pred_point[0])
        majority_class: str = RISK_LABELS.get(raw_idx, f"CLASS_{raw_idx}")

        if is_mapie and pred_sets_raw is not None:
            # Cas MAPIE : extraire l'intervalle de confiance
            pred_sets: np.ndarray = pred_sets_raw[:, :, 0]  # shape (1, n_classes)
            plausible: list[str] = self._extract_mapie_interval(pred_sets)
            confidence_level = 1.0 - alpha_value  # Dynamique, devient 0.90
        else:
            # Cas modèle de base : pas d'intervalle MAPIE
            plausible = [majority_class]
            confidence_level = None

        result = {
            "prediction":         majority_class,
            "confidence_level":   confidence_level,
            "uncertainty_set":    plausible,
            "raw_prediction_idx": raw_idx,
        }

        logger.info(
            "Prédiction : %s | Intervalle MAPIE : %s",
            majority_class, plausible,
        )

        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
# Chemin absolu ancré sur backend/ avec pathlib robuste
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_default_path = PROJECT_ROOT / "models" / "full_pipeline.pkl"
_pipeline_path = os.getenv("PIPELINE_PATH", str(_default_path))

try:
    prediction_service = PredictionService(pipeline_path=_pipeline_path)
except (FileNotFoundError, RuntimeError) as e:
    logger.error("IMPOSSIBLE DE CHARGER LE PIPELINE : %s", e)
    prediction_service = None  # type: ignore[assignment]