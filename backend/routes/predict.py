"""
routes/predict.py — Blueprint de Prédiction
============================================
Responsabilité UNIQUE : recevoir la requête HTTP, déléguer au service,
renvoyer la réponse JSON. Zéro logique métier ici.

Endpoint exposé :
  POST /api/predict
    Body  : JSON avec ~128 features brutes
    Return: {"status": "success", "prediction": str, "uncertainty_interval": list}
"""

import logging
from flask import Blueprint, request, jsonify, Response

from services.prediction_service import prediction_service

logger = logging.getLogger(__name__)

# ── Création du Blueprint ────────────────────────────────────────────────────
predict_bp = Blueprint("predict", __name__)


# ── POST /api/predict ────────────────────────────────────────────────────────

@predict_bp.post("/predict")
def predict() -> tuple[Response, int]:
    """
    Reçoit un payload JSON de features brutes et retourne la prédiction
    de risque aéronautique avec l'intervalle d'incertitude MAPIE.

    Request Body (application/json):
        {
            "ev_state": "TX",
            "weather": "VMC",
            "acft_make": "CESSNA",
            ... (~128 clés au total)
        }

    Success Response (200):
        {
            "status": "success",
            "prediction": "MINR",
            "confidence_level": 0.90,
            "uncertainty_interval": ["MINR", "SERS"]
        }

    Error Responses:
        400 — Payload manquant ou malformé
        503 — Pipeline ML non disponible
        500 — Erreur interne inattendue
    """

    # ── Garde-fou 1 : pipeline disponible ───────────────────────────────────
    if prediction_service is None:
        logger.error("Prédiction demandée mais le pipeline ML n'est pas chargé.")
        return jsonify({
            "status": "error",
            "code": "SERVICE_UNAVAILABLE",
            "message": (
                "Le pipeline ML n'est pas disponible. "
                "Vérifiez que 'models/full_pipeline.pkl' existe "
                "et que PIPELINE_PATH est correctement configuré."
            ),
        }), 503

    # ── Garde-fou 2 : Content-Type JSON ─────────────────────────────────────
    if not request.is_json:
        return jsonify({
            "status": "error",
            "code": "INVALID_CONTENT_TYPE",
            "message": "Le Content-Type doit être 'application/json'.",
        }), 400

    # ── Extraction du payload ────────────────────────────────────────────────
    payload: dict | None = request.get_json(silent=True)

    if not payload:
        return jsonify({
            "status": "error",
            "code": "EMPTY_PAYLOAD",
            "message": "Le corps de la requête est vide ou n'est pas un JSON valide.",
        }), 400

    if not isinstance(payload, dict):
        return jsonify({
            "status": "error",
            "code": "INVALID_PAYLOAD_TYPE",
            "message": "Le payload doit être un objet JSON (dict), pas un tableau.",
        }), 400

    # ── Délégation au service ────────────────────────────────────────────────
    try:
        result = prediction_service.predict_accident(payload)

    except ValueError as exc:
        # Erreur métier (payload invalide côté service)
        logger.warning("Payload invalide reçu : %s", exc)
        return jsonify({
            "status": "error",
            "code": "VALIDATION_ERROR",
            "message": str(exc),
        }), 400

    except RuntimeError as exc:
        # Erreur pipeline interne (ex: feature shape mismatch)
        logger.exception("Erreur runtime du pipeline ML : %s", exc)
        return jsonify({
            "status": "error",
            "code": "PIPELINE_ERROR",
            "message": (
                "Le pipeline ML a rencontré une erreur lors de la prédiction. "
                f"Détail : {exc}"
            ),
        }), 500

    except Exception as exc:
        # Erreur inattendue — on ne laisse jamais une exception remonter brute
        logger.exception("Erreur inattendue dans /api/predict : %s", exc)
        return jsonify({
            "status": "error",
            "code": "INTERNAL_ERROR",
            "message": "Une erreur interne inattendue s'est produite.",
        }), 500

    # ── Réponse de succès ────────────────────────────────────────────────────
    return jsonify({
        "status": "success",
        "prediction": result["prediction"],
        "confidence_level": result["confidence_level"],
        "uncertainty_interval": result["uncertainty_set"],
        # Le champ debug n'est exposé qu'en mode développement
        **({"_debug": {"raw_prediction_idx": result["raw_prediction_idx"]}}
           if _is_debug_mode() else {}),
    }), 200


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_debug_mode() -> bool:
    """Retourne True si Flask tourne en mode debug (env développement)."""
    import os
    return os.getenv("FLASK_DEBUG", "0") == "1"