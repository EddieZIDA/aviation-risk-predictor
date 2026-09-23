"""
routes/historical.py — Blueprint Données Historiques (EDA)
===========================================================
Responsabilité UNIQUE : exposer les données agrégées du dashboard d'exploration.
Toutes les requêtes passent par mongo_service, qui garantit l'exclusion
des données de test (split != "test").

Endpoints :
  GET /api/historical/stats                   → statistiques globales
  GET /api/historical/distribution?field=...  → effectifs par valeur d'un champ
  GET /api/historical/severity?field=...      → gravité croisée avec un champ
  GET /api/historical/timeseries              → accidents par année/mois
  GET /api/historical/risk-breakdown          → distribution NONE/MINR/SERS/FATL
  GET /api/historical/accidents               → échantillon de documents bruts
  GET /api/historical/random-example          → payload prêt pour /api/predict
"""

import logging
from collections.abc import Callable

from flask import Blueprint, Response, jsonify, request

from routes.errors import error_response, internal_error, service_unavailable
from services.mongo_service import mongo_service
from utils.feature_engineering import RAW_INPUT_COLS

logger = logging.getLogger(__name__)

historical_bp = Blueprint("historical", __name__)

# Liste blanche des champs agrégeables : empêche l'injection de noms de champs
# arbitraires (ex. "$where") dans le pipeline d'agrégation.
ALLOWED_FIELDS: set[str] = {
    "ev_state", "ev_year", "ev_month", "ev_season", "acft_make", "acft_category",
    "light_cond", "wx_cond_basic", "type_fly", "far_part", "num_eng",
    "crew_category", "sky_cond_ceil", "flt_plan_filed", "homebuilt",
}


def _query(fn: Callable[[], object], **extra) -> tuple[Response, int]:
    """Exécute une lecture MongoDB et l'enveloppe dans la réponse standard."""
    if mongo_service is None:
        return service_unavailable("MongoDB", "Vérifiez MONGO_URI et que le serveur est démarré.")
    try:
        return jsonify({"status": "success", **extra, "data": fn()}), 200
    except Exception:
        logger.exception("Erreur %s", request.path)
        return internal_error()


def _validated_field() -> tuple[str | None, tuple[Response, int] | None]:
    field = request.args.get("field", "").strip()
    if not field:
        return None, error_response(
            "MISSING_PARAM", "Le paramètre 'field' est requis (ex: ?field=ev_state).", 400)
    if field not in ALLOWED_FIELDS:
        return None, error_response(
            "INVALID_FIELD",
            f"Champ '{field}' non autorisé. Champs valides : {sorted(ALLOWED_FIELDS)}", 400)
    return field, None


@historical_bp.get("/historical/stats")
def get_stats() -> tuple[Response, int]:
    """{"data": {"total_accidents", "fatal_accidents", "fatal_rate", "years_covered", "states_count"}}"""
    return _query(lambda: mongo_service.get_stats_summary())


@historical_bp.get("/historical/distribution")
def get_distribution() -> tuple[Response, int]:
    """{"field": "ev_state", "data": [{"_id": "CA", "count": 312}, ...]}"""
    field, error = _validated_field()
    if error:
        return error
    return _query(lambda: mongo_service.get_distribution(field), field=field)


@historical_bp.get("/historical/severity")
def get_severity() -> tuple[Response, int]:
    """{"field": "light_cond", "data": [{"_id": "DAYL", "total": 900, "NONE": .., "FATL": ..}, ...]}"""
    field, error = _validated_field()
    if error:
        return error
    return _query(lambda: mongo_service.get_severity_by(field), field=field)


@historical_bp.get("/historical/timeseries")
def get_timeseries() -> tuple[Response, int]:
    """{"data": [{"year": 2010, "month": 1, "count": 45}, ...]}"""
    return _query(lambda: mongo_service.get_time_series())


@historical_bp.get("/historical/risk-breakdown")
def get_risk_breakdown() -> tuple[Response, int]:
    """{"data": [{"_id": "NONE", "count": 18200}, ...]}"""
    return _query(lambda: mongo_service.get_risk_breakdown())


@historical_bp.get("/historical/accidents")
def get_accidents_sample() -> tuple[Response, int]:
    """
    Query Params:
        state (str, optionnel) : code d'état US (ex: ?state=TX)
        limit (int, optionnel) : nombre de documents (défaut 200, max 500)
    """
    state = request.args.get("state", "").strip().upper() or None
    try:
        limit = max(1, min(int(request.args.get("limit", 200)), 500))
    except ValueError:
        limit = 200
    filters = {"ev_state": state} if state else None
    return _query(lambda: mongo_service.get_accidents_sample(filters=filters, limit=limit))


@historical_bp.get("/historical/random-example")
def get_random_example() -> tuple[Response, int]:
    """
    Document aléatoire (hors test) réduit aux colonnes du contrat d'entrée,
    prêt à être envoyé tel quel à POST /api/predict.
    """
    if mongo_service is None:
        return service_unavailable("MongoDB", "Vérifiez MONGO_URI et que le serveur est démarré.")
    try:
        doc = mongo_service.get_random_example(RAW_INPUT_COLS)
    except Exception:
        logger.exception("Erreur /historical/random-example")
        return internal_error()
    if doc is None:
        return error_response("EMPTY_COLLECTION", "La collection accidents est vide.", 404)
    return jsonify({"status": "success", "data": doc}), 200
