"""
app.py — Factory Flask
======================
Point d'entrée de l'application (pattern Application Factory).

Développement :  python app.py
Production    :  gunicorn -w 2 -b 0.0.0.0:5005 wsgi:app
"""

import logging

from flask import Flask, Response, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from config.settings import settings


def create_app() -> Flask:
    """Crée et configure l'instance Flask."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    )
    logging.getLogger("pymongo").setLevel(logging.WARNING)  # très verbeux en INFO

    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length
    app.json.ensure_ascii = False  # accents lisibles dans les réponses JSON

    # En dev, le proxy Vite rend CORS inutile ; il reste nécessaire si le
    # frontend est servi depuis une autre origine (VITE_API_URL absolue).
    CORS(app, resources={r"/api/*": {"origins": settings.frontend_urls}})

    _register_blueprints(app)
    _register_error_handlers(app)

    @app.get("/api/health")
    def system_health() -> tuple[Response, int]:
        """
        État réel des dépendances, affiché par la page Diagnostic du frontend.
        Répond toujours 200 : c'est le contenu qui indique ce qui est dégradé.
        """
        from services.gemini_service import gemini_service
        from services.mongo_service import mongo_service
        from services.prediction_service import prediction_service

        model = None
        if prediction_service is not None:
            model = {
                "name": prediction_service.model_name,
                "type": prediction_service.model_type,
                "confidence_level": prediction_service.confidence_level,
            }

        mongo_ok = mongo_service is not None and mongo_service.ping()
        return jsonify({
            "status": "online",
            "model_loaded": prediction_service is not None,
            "model": model,
            "mongodb": {
                "connected": mongo_ok,
                "test_split_protected": mongo_ok and mongo_service.has_split_field(),
            },
            "gemini_enabled": gemini_service is not None,
            "gemini_model": settings.gemini_model if gemini_service is not None else None,
        }), 200

    return app


def _register_blueprints(app: Flask) -> None:
    from routes.historical import historical_bp
    from routes.predict import predict_bp
    from routes.report import report_bp

    for blueprint in (predict_bp, report_bp, historical_bp):
        app.register_blueprint(blueprint, url_prefix="/api")


def _register_error_handlers(app: Flask) -> None:
    """Toutes les erreurs HTTP (404, 405, 413...) renvoient le format JSON de l'API."""

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException) -> tuple[Response, int]:
        return jsonify({
            "status": "error",
            "code": exc.name.upper().replace(" ", "_"),
            "message": exc.description,
        }), exc.code


if __name__ == "__main__":
    create_app().run(host=settings.host, port=settings.port, debug=settings.debug)
