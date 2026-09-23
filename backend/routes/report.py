"""
routes/report.py — Blueprint de Génération de Rapports
========================================================
Responsabilité UNIQUE : valider l'entrée, déléguer à gemini_service,
retourner le rapport structuré. Zéro logique LLM ici.

Endpoint :
  POST /api/report
    Body  : { prediction, uncertainty_interval, accident_features }
    Return: { status, report: { risk_summary, contributing_factors, ... } }
"""

import logging

from flask import Blueprint, Response, jsonify, request

from routes.errors import error_response, internal_error, read_json_object, service_unavailable
from services.gemini_service import gemini_service

logger = logging.getLogger(__name__)

report_bp = Blueprint("report", __name__)


@report_bp.post("/report")
def generate_report() -> tuple[Response, int]:
    """
    Request Body (application/json):
        {
            "prediction": "MINR",
            "uncertainty_interval": ["MINR", "SERS"],
            "accident_features": {"ev_state": "TX", "acft_make": "CESSNA", ...}
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
                "_meta": {...}
            }
        }

    Error Responses:
        400 — entrée invalide
        502 — Gemini injoignable ou réponse inexploitable
        503 — clé API absente
    """
    if gemini_service is None:
        return service_unavailable("Le service Gemini", "Vérifiez GEMINI_API_KEY dans backend/.env.")

    body, error = read_json_object(request)
    if error:
        return error

    prediction = body.get("prediction")
    uncertainty_interval = body.get("uncertainty_interval")
    accident_features = body.get("accident_features")

    if not isinstance(prediction, str) or not prediction.strip():
        return error_response("INVALID_PREDICTION", "'prediction' doit être une chaîne non vide.", 400)
    if not isinstance(uncertainty_interval, list):
        return error_response("INVALID_UNCERTAINTY", "'uncertainty_interval' doit être une liste.", 400)
    if not isinstance(accident_features, dict):
        return error_response("INVALID_FEATURES", "'accident_features' doit être un objet JSON.", 400)

    try:
        report = gemini_service.generate_safety_report(
            prediction=prediction,
            uncertainty_set=uncertainty_interval,
            accident_features=accident_features,
        )
    except ValueError as exc:
        return error_response("VALIDATION_ERROR", str(exc), 400)
    except RuntimeError as exc:
        logger.error("Erreur Gemini : %s", exc)
        return error_response(
            "GEMINI_API_ERROR",
            "Le service de génération de rapport n'a pas pu répondre. Réessayez dans un instant.",
            502,
        )
    except Exception:
        logger.exception("Erreur inattendue dans /api/report")
        return internal_error()

    return jsonify({"status": "success", "report": report}), 200
