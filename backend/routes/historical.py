"""
routes/historical.py — Blueprint Données Historiques (EDA)
===========================================================
Responsabilité UNIQUE : exposer les données agrégées pour le dashboard
d'exploration. Toutes les requêtes passent par mongo_service qui garantit
l'exclusion des données test (split != "test").

Endpoints exposés :
  GET /api/historical/stats          → Statistiques globales
  GET /api/historical/distribution   → Distribution par champ (query param)
  GET /api/historical/timeseries     → Série temporelle mensuelle
  GET /api/historical/risk-breakdown → Distribution des niveaux de risque
  GET /api/historical/accidents      → Échantillon de données brutes
"""

import logging
from flask import Blueprint, request, jsonify, Response

from services.mongo_service import mongo_service

logger = logging.getLogger(__name__)

historical_bp = Blueprint("historical", __name__)

# Champs autorisés pour la distribution (whitelist — évite les injections)
ALLOWED_DISTRIBUTION_FIELDS: set[str] = {
    "ev_state", "ev_type", "acft_make", "acft_category",
    "phase_flt_spec", "weather", "light_cond", "damage",
    "type_fly", "far_part", "num_eng", "ev_month", "ev_year",
}


# ── GET /api/historical/stats ─────────────────────────────────────────────────

@historical_bp.get("/historical/stats")
def get_stats() -> tuple[Response, int]:
    """
    Retourne les statistiques globales du dataset (hors test).

    Success Response (200):
        {
            "status": "success",
            "data": {
                "total_accidents": 42000,
                "years_covered": [1982, ..., 2023],
                "states_count": 51
            }
        }
    """
    if mongo_service is None:
        return _mongo_unavailable_response()

    try:
        stats = mongo_service.get_stats_summary()
        return jsonify({"status": "success", "data": stats}), 200

    except Exception as exc:
        logger.exception("Erreur /historical/stats : %s", exc)
        return _internal_error_response(exc)


# ── GET /api/historical/distribution ─────────────────────────────────────────

@historical_bp.get("/historical/distribution")
def get_distribution() -> tuple[Response, int]:
    """
    Retourne la distribution (count) d'un champ donné.
    Le champ est passé en query parameter `field`.

    Query Params:
        field (str, requis) : Champ à agréger. Doit être dans ALLOWED_DISTRIBUTION_FIELDS.

    Success Response (200):
        {
            "status": "success",
            "field": "ev_state",
            "data": [{"_id": "CA", "count": 312}, ...]
        }
    """
    if mongo_service is None:
        return _mongo_unavailable_response()

    field = request.args.get("field", "").strip()

    if not field:
        return jsonify({
            "status": "error",
            "code": "MISSING_PARAM",
            "message": "Le paramètre 'field' est requis (ex: ?field=ev_state).",
        }), 400

    # Whitelist : protection contre les injections de champs arbitraires
    if field not in ALLOWED_DISTRIBUTION_FIELDS:
        return jsonify({
            "status": "error",
            "code": "INVALID_FIELD",
            "message": (
                f"Le champ '{field}' n'est pas autorisé. "
                f"Champs valides : {sorted(ALLOWED_DISTRIBUTION_FIELDS)}"
            ),
        }), 400

    try:
        data = mongo_service.get_distribution(group_field=field)
        return jsonify({
            "status": "success",
            "field": field,
            "data": data,
        }), 200

    except Exception as exc:
        logger.exception("Erreur /historical/distribution?field=%s : %s", field, exc)
        return _internal_error_response(exc)


# ── GET /api/historical/timeseries ───────────────────────────────────────────

@historical_bp.get("/historical/timeseries")
def get_timeseries() -> tuple[Response, int]:
    """
    Retourne la série temporelle mensuelle des accidents (hors test).

    Success Response (200):
        {
            "status": "success",
            "data": [
                {"year": 2010, "month": 1, "count": 45},
                ...
            ]
        }
    """
    if mongo_service is None:
        return _mongo_unavailable_response()

    try:
        data = mongo_service.get_time_series()
        return jsonify({"status": "success", "data": data}), 200

    except Exception as exc:
        logger.exception("Erreur /historical/timeseries : %s", exc)
        return _internal_error_response(exc)


# ── GET /api/historical/risk-breakdown ───────────────────────────────────────

@historical_bp.get("/historical/risk-breakdown")
def get_risk_breakdown() -> tuple[Response, int]:
    """
    Retourne la distribution des niveaux de risque (FATL/SERS/MINR/NONE).

    Success Response (200):
        {
            "status": "success",
            "data": [
                {"_id": "NONE", "count": 18200},
                {"_id": "MINR", "count": 12500},
                ...
            ]
        }
    """
    if mongo_service is None:
        return _mongo_unavailable_response()

    try:
        data = mongo_service.get_risk_breakdown()
        return jsonify({"status": "success", "data": data}), 200

    except Exception as exc:
        logger.exception("Erreur /historical/risk-breakdown : %s", exc)
        return _internal_error_response(exc)


# ── GET /api/historical/accidents ────────────────────────────────────────────

@historical_bp.get("/historical/accidents")
def get_accidents_sample() -> tuple[Response, int]:
    """
    Retourne un échantillon de données brutes (hors test).
    Supporte un filtre optionnel par état (`state` query param).

    Query Params:
        state (str, optionnel) : Code d'état US pour filtrer (ex: ?state=TX)
        limit (int, optionnel) : Nombre de résultats (défaut: 200, max: 500)

    Success Response (200):
        {
            "status": "success",
            "count": 200,
            "data": [{ ... }, ...]
        }
    """
    if mongo_service is None:
        return _mongo_unavailable_response()

    # Parsing des query params
    state = request.args.get("state", "").strip().upper() or None
    try:
        limit = min(int(request.args.get("limit", 200)), 500)
    except (ValueError, TypeError):
        limit = 200

    filters = {"ev_state": state} if state else None

    try:
        data = mongo_service.get_accidents_sample(filters=filters, limit=limit)
        return jsonify({
            "status": "success",
            "count": len(data),
            "data": data,
        }), 200

    except Exception as exc:
        logger.exception("Erreur /historical/accidents : %s", exc)
        return _internal_error_response(exc)


# ── Helpers de réponse d'erreur ───────────────────────────────────────────────

def _mongo_unavailable_response() -> tuple[Response, int]:
    return jsonify({
        "status": "error",
        "code": "SERVICE_UNAVAILABLE",
        "message": (
            "MongoDB n'est pas disponible. "
            "Vérifiez MONGO_URI et MONGO_DB_NAME dans votre .env."
        ),
    }), 503


def _internal_error_response(exc: Exception) -> tuple[Response, int]:
    return jsonify({
        "status": "error",
        "code": "INTERNAL_ERROR",
        "message": f"Erreur interne : {exc}",
    }), 500


# ── GET /api/historical/random-example ───────────────────────────────────────

@historical_bp.get("/historical/random-example")
def get_random_example() -> tuple[Response, int]:
    """
    Retourne un document aléatoire de la collection accidents (hors test)
    formaté comme payload brut prêt à être collé dans le formulaire frontend.

    Le tirage est fait via $sample MongoDB — vraiment aléatoire, efficace
    même sur des collections de 50 000+ documents.

    Success Response (200):
        {
            "status": "success",
            "data": {
                "ev_state": "TX",
                "ev_year": 2019,
                "wx_temp": 85.0,
                ...  (uniquement les colonnes RAW_INPUT_COLS non-nulles)
            }
        }
    """
    if mongo_service is None:
        return _mongo_unavailable_response()

    try:
        doc = mongo_service.get_random_example()

        if doc is None:
            return jsonify({
                "status": "error",
                "code": "EMPTY_COLLECTION",
                "message": "La collection accidents est vide ou inaccessible.",
            }), 404

        return jsonify({"status": "success", "data": doc}), 200

    except Exception as exc:
        logger.exception("Erreur /historical/random-example : %s", exc)
        return _internal_error_response(exc)