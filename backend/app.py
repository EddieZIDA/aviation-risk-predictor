"""
app.py — Factory Flask
======================
Point d'entrée de l'application. Utilise le pattern Application Factory.
"""

import os
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

def create_app() -> Flask:
    """
    Crée et configure l'instance Flask.
    """
    app = Flask(__name__)

    # ── Configuration ────────────────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-fallback")
    
    # Calcul du PROJECT_ROOT : backend/app.py -> backend -> racine
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _default_pipeline = PROJECT_ROOT / "models" / "full_pipeline.pkl"
    app.config["PIPELINE_PATH"] = os.getenv("PIPELINE_PATH", str(_default_pipeline))

    # ── CORS ────────────────────────────────────────────────────────────────
    # Autorise les requêtes depuis le frontend (par défaut port 5173 pour Vite)
    allowed_origins = os.getenv("FRONTEND_URL", "http://localhost:5173")
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

    # ── Enregistrement des Blueprints ────────────────────────────────────
    _register_blueprints(app)

    # ── Route de Diagnostic Système ──────────────────────────────────────
    @app.get("/api/health")
    def system_health():
        """
        Retourne l'état complet des artefacts et services.
        Répond aux appels de la page HealthPage du frontend.
        """
        from services.mongo_service import mongo_service
        from services.prediction_service import prediction_service

        # Vérification dynamique de l'état
        status = {
            "status": "online",
            "model_loaded": False,
            "preprocessor_loaded": False,
            "mongodb_config": {
                "uri_set": False,
                "collection": None
            },
            "gemini_enabled": bool(os.getenv("GEMINI_API_KEY"))
        }

        if prediction_service:
            status["model_loaded"] = hasattr(prediction_service, '_model') and prediction_service._model is not None
            status["preprocessor_loaded"] = hasattr(prediction_service, '_preprocessor') and prediction_service._preprocessor is not None

        if mongo_service:
            status["mongodb_config"]["uri_set"] = True
            status["mongodb_config"]["collection"] = "accidents"

        return jsonify(status), 200

    return app


def _register_blueprints(app: Flask) -> None:
    """Centralise l'enregistrement des routes."""
    from routes.predict import predict_bp
    from routes.report import report_bp
    from routes.historical import historical_bp
    
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(report_bp, url_prefix="/api")
    app.register_blueprint(historical_bp, url_prefix="/api")


# ── Lancement ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    flask_app = create_app()
    # Utilisation du port 5005 comme spécifié dans les logs système
    port = int(os.getenv("PORT", 5005))
    
    flask_app.run(
        host="0.0.0.0",
        port=port,
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )