"""
routes/predict.py — Blueprint de Prédiction
============================================
Responsabilité UNIQUE : recevoir la requête HTTP, déléguer au service,
renvoyer la réponse JSON. Zéro logique métier ici.

Endpoints :
  POST /api/predict        → prédiction + ensemble MAPIE
  GET  /api/predict/schema → colonnes attendues (contrat d'entrée)
"""

import logging

from flask import Blueprint, Response, jsonify, request

from config.settings import settings
from routes.errors import error_response, internal_error, read_json_object, service_unavailable
from services.prediction_service import prediction_service
from utils.feature_engineering import (
    BOOL_INPUT_COLS,
    CATEGORICAL_INPUT_COLS,
    NUMERIC_INPUT_COLS,
    RAW_INPUT_COLS,
)

logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict", __name__)


@predict_bp.post("/predict")
def predict() -> tuple[Response, int]:
    """
    Request Body (application/json) — une ligne au format du CSV nettoyé
    (clés de RAW_INPUT_COLS ; les clés absentes sont imputées) :
        {"ev_state": "TX", "acft_make": "CESSNA", "wx_temp_c": 21.0, ...}

    Success Response (200):
        {
            "status": "success",
            "prediction": "MINR",
            "confidence_level": 0.90,
            "uncertainty_interval": ["MINR", "SERS"],
            "probabilities": {"NONE": 0.21, "MINR": 0.45, "SERS": 0.2, "FATL": 0.14},
            "model": {"name": "XGBClassifier", "type": "MAPIE (LAC)"},
            "input_coverage": {"provided": 42, "expected": 64}
        }

    Error Responses:
        400 — payload manquant, malformé ou valeur mal typée
        503 — pipeline ML non chargé
        500 — erreur interne
    """
    if prediction_service is None:
        return service_unavailable(
            "Le pipeline ML", "Vérifiez que les notebooks 03 à 05 ont produit leurs artefacts.")

    payload, error = read_json_object(request)
    if error:
        return error

    try:
        result = prediction_service.predict_accident(payload)
    except ValueError as exc:
        logger.warning("Payload invalide : %s", exc)
        return error_response("VALIDATION_ERROR", str(exc), 400)
    except Exception:
        logger.exception("Erreur dans /api/predict")
        return internal_error()

    provided = sum(1 for col in RAW_INPUT_COLS if payload.get(col) not in (None, ""))
    response = {
        "status": "success",
        "prediction": result["prediction"],
        "confidence_level": result["confidence_level"],
        "uncertainty_interval": result["uncertainty_set"],
        "probabilities": result["probabilities"],
        "model": {"name": prediction_service.model_name, "type": prediction_service.model_type},
        "input_coverage": {"provided": provided, "expected": len(RAW_INPUT_COLS)},
    }
    if settings.debug:
        response["_debug"] = {"raw_prediction_idx": result["raw_prediction_idx"]}
    return jsonify(response), 200


@predict_bp.get("/predict/schema")
def predict_schema() -> tuple[Response, int]:
    """Contrat d'entrée de /api/predict, groupé par type de valeur."""
    return jsonify({
        "status": "success",
        "fields": {
            "numeric": NUMERIC_INPUT_COLS,
            "boolean": BOOL_INPUT_COLS,
            "categorical": CATEGORICAL_INPUT_COLS,
        },
    }), 200
