"""
routes/report.py — Blueprint de Génération de Rapports
========================================================
Responsabilité UNIQUE : valider l'entrée, déléguer à gemini_service,
retourner le rapport structuré. Zéro logique LLM ici.

Endpoint exposé :
  POST /api/report
    Body  : { prediction, uncertainty_interval, accident_features }
    Return: { status, report: { risk_summary, contributing_factors, ... } }
"""

import logging
from flask import Blueprint, request, jsonify, Response

from services.gemini_service import gemini_service

logger = logging.getLogger(__name__)

report_bp = Blueprint("report", __name__)


# ── POST /api/report ─────────────────────────────────────────────────────────

@report_bp.post("/report")
def generate_report() -> tuple[Response, int]:
    """
    Reçoit la prédiction MAPIE et les features, retourne un rapport
    de sécurité structuré généré par Gemini.

    Request Body (application/json):
        {
            "prediction": "MINR",
            "uncertainty_interval": ["MINR", "SERS"],
            "accident_features": {
                "ev_state": "TX",
                "acft_make": "CESSNA",
                ...
            }
        }

    Success Response (200):
        {
            "status": "success",
            "report": {
                "risk_summary": "...",
                "contributing_factors": [...],
                "safety_recommendations": [...],
                "worst_case_preparedness": "...",
                "confidence_note": "...",
                "_meta": { ... }
            }
        }
    """

    # ── Garde-fou 1 : service disponible ─────────────────────────────────
    if gemini_service is None:
        return jsonify({
            "status": "error",
            "code": "SERVICE_UNAVAILABLE",
            "message": (
                "Le service Gemini n'est pas disponible. "
                "Vérifiez GEMINI_API_KEY dans votre .env."
            ),
        }), 503

    # ── Garde-fou 2 : Content-Type ────────────────────────────────────────
    if not request.is_json:
        return jsonify({
            "status": "error",
            "code": "INVALID_CONTENT_TYPE",
            "message": "Le Content-Type doit être 'application/json'.",
        }), 400

    body: dict | None = request.get_json(silent=True)

    if not body or not isinstance(body, dict):
        return jsonify({
            "status": "error",
            "code": "EMPTY_PAYLOAD",
            "message": "Le corps de la requête est vide ou invalide.",
        }), 400

    # ── Validation des champs requis ──────────────────────────────────────
    missing = [
        field for field in ("prediction", "uncertainty_interval", "accident_features")
        if field not in body
    ]
    if missing:
        return jsonify({
            "status": "error",
            "code": "MISSING_FIELDS",
            "message": f"Champs obligatoires manquants : {missing}",
        }), 400

    prediction: str            = body["prediction"]
    uncertainty_interval: list = body["uncertainty_interval"]
    accident_features: dict    = body["accident_features"]

    # Validations de type minimales
    if not isinstance(prediction, str) or not prediction.strip():
        return jsonify({
            "status": "error",
            "code": "INVALID_PREDICTION",
            "message": "'prediction' doit être une chaîne non vide.",
        }), 400

    if not isinstance(uncertainty_interval, list):
        return jsonify({
            "status": "error",
            "code": "INVALID_UNCERTAINTY",
            "message": "'uncertainty_interval' doit être une liste.",
        }), 400

    if not isinstance(accident_features, dict):
        return jsonify({
            "status": "error",
            "code": "INVALID_FEATURES",
            "message": "'accident_features' doit être un objet JSON.",
        }), 400

    # ── Délégation au service ─────────────────────────────────────────────
    try:
        report = gemini_service.generate_safety_report(
            prediction=prediction,
            uncertainty_set=uncertainty_interval,
            accident_features=accident_features,
        )

    except ValueError as exc:
        logger.warning("Erreur de validation Gemini service : %s", exc)
        return jsonify({
            "status": "error",
            "code": "VALIDATION_ERROR",
            "message": str(exc),
        }), 400

    except RuntimeError as exc:
        logger.exception("Erreur API Gemini : %s", exc)
        return jsonify({
            "status": "error",
            "code": "GEMINI_API_ERROR",
            "message": f"Erreur lors de la génération du rapport : {exc}",
        }), 502  # 502 Bad Gateway = erreur du service externe

    except Exception as exc:
        logger.exception("Erreur inattendue dans /api/report : %s", exc)
        return jsonify({
            "status": "error",
            "code": "INTERNAL_ERROR",
            "message": "Une erreur interne inattendue s'est produite.",
        }), 500

    # ── Réponse de succès ─────────────────────────────────────────────────
    return jsonify({
        "status": "success",
        "report": report,
    }), 200