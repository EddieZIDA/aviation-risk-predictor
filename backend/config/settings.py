"""
config/settings.py — Configuration centralisée
===============================================
Toutes les variables d'environnement sont lues ICI et nulle part ailleurs.
Le fichier backend/.env est chargé automatiquement (voir .env.example).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(int(default))).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    # Flask
    secret_key: str = os.getenv("SECRET_KEY", "dev-only-secret")
    debug: bool = _env_bool("FLASK_DEBUG")
    host: str = os.getenv("FLASK_HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "5005"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # Taille maximale d'un corps de requête (un payload fait ~5 Ko)
    max_content_length: int = 256 * 1024

    # CORS : liste d'origines séparées par des virgules
    frontend_urls: list[str] = field(default_factory=lambda: [
        url.strip() for url in os.getenv("FRONTEND_URL", "http://localhost:5173").split(",")
    ])

    # Artefacts ML (sorties des notebooks 03 à 05)
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "notebooks")))
    mapie_alpha: float = float(os.getenv("MAPIE_ALPHA", "0.10"))

    # MongoDB
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    mongo_db_name: str = os.getenv("MONGO_DB_NAME", "aviation_risk")

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    gemini_timeout_s: float = float(os.getenv("GEMINI_TIMEOUT_S", "30"))
    # Utilisé si le modèle principal reste surchargé (503) ou hors quota (429) ; vide = désactivé
    gemini_fallback_model: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")


settings = Settings()

if not settings.debug and settings.secret_key == "dev-only-secret":
    import logging
    logging.getLogger(__name__).warning("SECRET_KEY par défaut utilisée : à définir en production.")
