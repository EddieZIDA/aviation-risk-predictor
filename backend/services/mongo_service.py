"""
services/mongo_service.py — Service MongoDB
============================================
Responsabilité UNIQUE : toutes les interactions avec MongoDB.
Ce service est le SEUL à connaître l'existence de pymongo.

RÈGLE D'OR — ZÉRO DATA LEAKAGE :
  Chaque requête intègre OBLIGATOIREMENT le filtre `{"split": {"$ne": "test"}}`
  (champ écrit par seed_mongo.py avec le découpage exact du NB03).
  Ce filtre est injecté par `_safe_filter()` / `_safe_match()` et ne peut pas
  être contourné par l'appelant.
"""

import logging
import math
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from config.settings import settings

logger = logging.getLogger(__name__)

ACCIDENTS = "accidents"

# Seule source de vérité pour l'exclusion des données de test.
LEAKAGE_GUARD: dict = {"split": {"$ne": "test"}}

# Valeurs considérées comme « non renseignées » dans les agrégations.
EMPTY_VALUES: list = [None, "", "null", "NaN", "Unknown", "unknown", "UNK"]


class MongoService:
    """Accès en lecture seule à la collection `accidents` (hors données de test)."""

    def __init__(self, uri: str, db_name: str) -> None:
        """
        Raises:
            PyMongoError: si le serveur est injoignable (timeout 5 s).
        """
        self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self._client.admin.command("ping")
        self._db = self._client[db_name]
        logger.info("MongoDB connecté — base '%s'", db_name)

    # ── Utilitaires ──────────────────────────────────────────────────────────

    @property
    def _accidents(self) -> Collection:
        return self._db[ACCIDENTS]

    @staticmethod
    def _safe_filter(extra_filter: dict | None = None) -> dict:
        """
        Fusionne LEAKAGE_GUARD avec un filtre métier.

        Example:
            _safe_filter({"ev_state": "TX"})
            → {"split": {"$ne": "test"}, "ev_state": "TX"}

        Raises:
            ValueError: si l'appelant tente de filtrer lui-même sur `split`.
        """
        if extra_filter and "split" in extra_filter:
            raise ValueError("Le champ 'split' est réservé au garde-fou anti-fuite.")
        return {**LEAKAGE_GUARD, **(extra_filter or {})}

    def ping(self) -> bool:
        """True si le serveur répond (utilisé par /api/health)."""
        try:
            self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    def has_split_field(self) -> bool:
        """False si la collection a été chargée sans le champ `split` (ancien seed)."""
        return self._accidents.find_one({"split": {"$exists": True}}, {"_id": 1}) is not None

    # ── Lectures ─────────────────────────────────────────────────────────────

    def get_accidents_sample(self, filters: dict | None = None, limit: int = 500) -> list[dict]:
        """Échantillon de documents bruts (sans _id ni split)."""
        cursor = self._accidents.find(
            self._safe_filter(filters), {"_id": 0, "split": 0}
        ).limit(limit)
        return [_json_safe(doc) for doc in cursor]

    def get_distribution(self, group_field: str, limit: int = 100) -> list[dict]:
        """
        Nombre d'accidents par valeur d'un champ, trié par fréquence décroissante.
        Le nom du champ DOIT avoir été validé par l'appelant (liste blanche).

        Returns:
            [{"_id": "CA", "count": 312}, ...]
        """
        pipeline = [
            {"$match": {**LEAKAGE_GUARD, group_field: {"$nin": EMPTY_VALUES}}},
            {"$group": {"_id": f"${group_field}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return list(self._accidents.aggregate(pipeline))

    def get_time_series(self) -> list[dict]:
        """Nombre d'accidents par (année, mois), trié chronologiquement."""
        pipeline = [
            {"$match": LEAKAGE_GUARD},
            {"$group": {"_id": {"year": "$ev_year", "month": "$ev_month"}, "count": {"$sum": 1}}},
            {"$sort": {"_id.year": 1, "_id.month": 1}},
            {"$project": {"_id": 0, "year": "$_id.year", "month": "$_id.month", "count": 1}},
        ]
        return list(self._accidents.aggregate(pipeline))

    def get_risk_breakdown(self) -> list[dict]:
        """Distribution de la cible ev_highest_injury (NONE/MINR/SERS/FATL)."""
        return self.get_distribution("ev_highest_injury")

    def get_severity_by(self, group_field: str, limit: int = 15) -> list[dict]:
        """
        Répartition de la gravité pour les `limit` valeurs les plus fréquentes
        d'un champ. Le nom du champ DOIT avoir été validé par l'appelant.

        Returns:
            [{"_id": "IMC", "total": 812, "NONE": 300, "MINR": 90, "SERS": 80, "FATL": 342}, ...]
        """
        pipeline = [
            {"$match": {**LEAKAGE_GUARD, group_field: {"$nin": EMPTY_VALUES}}},
            {"$group": {
                "_id": f"${group_field}",
                "total": {"$sum": 1},
                **{level: {"$sum": {"$cond": [{"$eq": ["$ev_highest_injury", level]}, 1, 0]}}
                   for level in ("NONE", "MINR", "SERS", "FATL")},
            }},
            {"$sort": {"total": -1}},
            {"$limit": limit},
        ]
        return list(self._accidents.aggregate(pipeline))

    def get_stats_summary(self) -> dict[str, Any]:
        """Statistiques globales pour l'en-tête du dashboard."""
        safe = self._safe_filter()
        years = [y for y in self._accidents.distinct("ev_year", safe) if y]
        states = [s for s in self._accidents.distinct("ev_state", safe) if s]
        fatal = self._accidents.count_documents(self._safe_filter({"ev_highest_injury": "FATL"}))
        total = self._accidents.count_documents(safe)
        return {
            "total_accidents": total,
            "fatal_accidents": fatal,
            "fatal_rate": round(fatal / total, 4) if total else 0.0,
            "years_covered": sorted(years),
            "states_count": len(states),
        }

    def get_random_example(self, fields: list[str]) -> dict | None:
        """
        Document aléatoire (hors test) réduit aux `fields` demandés et
        sans valeurs nulles — prêt à être envoyé à POST /api/predict.
        """
        pipeline = [
            {"$match": LEAKAGE_GUARD},
            {"$sample": {"size": 1}},
            {"$project": {"_id": 0, **{f: 1 for f in fields}}},
        ]
        docs = list(self._accidents.aggregate(pipeline))
        if not docs:
            return None
        return {k: v for k, v in _json_safe(docs[0]).items() if v is not None}

    def close(self) -> None:
        self._client.close()


def _json_safe(doc: dict) -> dict:
    """Remplace NaN/±inf (non sérialisables en JSON standard) par None."""
    return {
        k: None if isinstance(v, float) and not math.isfinite(v) else v
        for k, v in doc.items()
    }


# ── Instance singleton ────────────────────────────────────────────────────────
try:
    mongo_service: MongoService | None = MongoService(settings.mongo_uri, settings.mongo_db_name)
except Exception as e:  # serveur éteint, URI invalide...
    logger.error("IMPOSSIBLE DE CONNECTER MONGODB : %s", e)
    mongo_service = None
