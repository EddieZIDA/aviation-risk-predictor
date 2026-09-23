"""
routes/errors.py — Réponses d'erreur JSON uniformes
====================================================
Format commun à toute l'API :
    {"status": "error", "code": "<CODE_MACHINE>", "message": "<texte lisible>"}

Les messages des erreurs 5xx ne contiennent jamais le détail de l'exception
(qui peut révéler des chemins ou de la configuration) : ce détail va dans les logs.
"""

from flask import Response, jsonify


def error_response(code: str, message: str, status: int) -> tuple[Response, int]:
    return jsonify({"status": "error", "code": code, "message": message}), status


def service_unavailable(service: str, hint: str) -> tuple[Response, int]:
    return error_response("SERVICE_UNAVAILABLE", f"{service} n'est pas disponible. {hint}", 503)


def internal_error() -> tuple[Response, int]:
    return error_response(
        "INTERNAL_ERROR", "Une erreur interne s'est produite. Consultez les logs du serveur.", 500
    )


def read_json_object(request) -> tuple[dict | None, tuple[Response, int] | None]:
    """
    Extrait un objet JSON du corps de la requête.
    Retourne (payload, None) si valide, sinon (None, réponse d'erreur 400).
    """
    if not request.is_json:
        return None, error_response(
            "INVALID_CONTENT_TYPE", "Le Content-Type doit être 'application/json'.", 400)
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not body:
        return None, error_response(
            "INVALID_PAYLOAD", "Le corps doit être un objet JSON non vide.", 400)
    return body, None
