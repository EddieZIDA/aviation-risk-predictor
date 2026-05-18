"""
services/mongo_service.py — Service MongoDB
============================================
Responsabilité UNIQUE : toutes les interactions avec MongoDB.
Ce service est le SEUL à connaître l'existence de pymongo.

RÈGLE D'OR — ZÉRO DATA LEAKAGE :
  Chaque méthode `find()` de ce service intègre OBLIGATOIREMENT
  le filtre `{"split": {"$ne": "test"}}`.
  Ce filtre n'est JAMAIS optionnel, JAMAIS contournable par le caller.
  Il est encapsulé dans la constante LEAKAGE_GUARD et injecté
  via `_safe_filter()` avant CHAQUE requête.
"""

import os
import logging
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure

logger = logging.getLogger(__name__)

# ── Garde-fou anti data leakage ───────────────────────────────────────────────
# Cette constante est la seule source de vérité pour l'exclusion des données test.
# Elle est INJECTÉE dans chaque requête find() via _safe_filter().
LEAKAGE_GUARD: dict = {"split": {"$ne": "test"}}


class MongoService:
    """
    Service singleton pour toutes les opérations MongoDB.
    Garantit l'exclusion systématique des données de test (split == 'test').
    """

    def __init__(self, uri: str, db_name: str) -> None:
        """
        Initialise la connexion MongoDB.

        Args:
            uri    : URI de connexion MongoDB (ex: 'mongodb://localhost:27017/')
            db_name: Nom de la base de données (ex: 'aviation_risk')

        Raises:
            ConnectionFailure: Si la connexion au serveur échoue.
            RuntimeError     : Pour toute autre erreur d'initialisation.
        """
        try:
            self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Ping pour valider la connexion dès l'init
            self._client.admin.command("ping")
            self._db = self._client[db_name]
            logger.info(
                "MongoDB connecté — base : '%s' @ %s", db_name, uri
            )
        except ConnectionFailure as exc:
            raise ConnectionFailure(
                f"Impossible de se connecter à MongoDB ({uri}) : {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Erreur d'initialisation MongoDB : {exc}"
            ) from exc

    # ── Collection accessor ───────────────────────────────────────────────────

    def _get_collection(self, name: str) -> Collection:
        """Retourne une collection MongoDB par son nom."""
        return self._db[name]

    # ── Garde-fou anti data leakage ───────────────────────────────────────────

    @staticmethod
    def _safe_filter(extra_filter: dict | None = None) -> dict:
        """
        Fusionne LEAKAGE_GUARD avec tout filtre additionnel.

        Cette méthode est appelée AVANT chaque find(). Elle garantit
        que le filtre `split != "test"` est TOUJOURS présent.

        Args:
            extra_filter: Filtres métier optionnels du caller.

        Returns:
            Filtre MongoDB fusionné, incluant toujours LEAKAGE_GUARD.

        Example:
            _safe_filter({"ev_state": "TX"})
            → {"split": {"$ne": "test"}, "ev_state": "TX"}
        """
        base = dict(LEAKAGE_GUARD)  # Copie défensive
        if extra_filter:
            base.update(extra_filter)
        return base

    # ── Méthodes de lecture ───────────────────────────────────────────────────

    def get_accidents_sample(
        self,
        filters: dict | None = None,
        limit: int = 500,
        projection: dict | None = None,
    ) -> list[dict]:
        """
        Retourne un échantillon d'accidents (hors données test).

        Args:
            filters   : Filtres métier additionnels (ex: {"ev_state": "TX"})
            limit     : Nombre maximum de documents retournés (défaut: 500)
            projection: Champs à inclure/exclure (ex: {"_id": 0, "ev_state": 1})

        Returns:
            Liste de documents MongoDB sérialisables en JSON.
        """
        collection = self._get_collection("accidents")
        safe_filter = self._safe_filter(filters)

        # Projection par défaut : on exclut _id (non sérialisable en JSON)
        if projection is None:
            projection = {"_id": 0}
        else:
            projection.setdefault("_id", 0)

        try:
            cursor = collection.find(safe_filter, projection).limit(limit)
            return list(cursor)
        except OperationFailure as exc:
            logger.error("Échec requête MongoDB (accidents) : %s", exc)
            raise

    def get_distribution(
        self,
        group_field: str,
        collection_name: str = "accidents",
    ) -> list[dict]:
        """
        Agrégation de distribution : compte les accidents par valeur d'un champ.
        Utile pour les graphiques EDA (ex: distribution par état, par phase de vol).

        Toujours filtré hors données test via LEAKAGE_GUARD.

        Args:
            group_field     : Nom du champ pour le GROUP BY (ex: "ev_state")
            collection_name : Nom de la collection (défaut: "accidents")

        Returns:
            Liste de dicts [{_id: valeur, count: int}] triée par count décroissant.

        Example:
            get_distribution("ev_state")
            → [{"_id": "CA", "count": 312}, {"_id": "TX", "count": 287}, ...]
        """
        collection = self._get_collection(collection_name)

        pipeline = [
            # Étape 1 : exclure les données test ET les valeurs vides — OBLIGATOIRE
            {"$match": {
                **LEAKAGE_GUARD,
                group_field: {"$nin": [None, "", "null", "NaN", "Unknown", "unknown"]}
            }},
            # Étape 2 : grouper et compter
            {"$group": {"_id": f"${group_field}", "count": {"$sum": 1}}},
            # Étape 3 : trier par fréquence décroissante
            {"$sort": {"count": -1}},
            # Étape 4 : limiter pour éviter des payloads trop lourds
            {"$limit": 100}
        ]

        try:
            return list(collection.aggregate(pipeline))
        except OperationFailure as exc:
            logger.error(
                "Échec agrégation MongoDB (field=%s) : %s", group_field, exc
            )
            raise

    def get_time_series(
        self,
        year_field: str = "ev_year",
        month_field: str = "ev_month",
    ) -> list[dict]:
        """
        Agrégation temporelle : nombre d'accidents par année/mois.
        Utilisée pour la courbe temporelle du dashboard EDA.

        Args:
            year_field : Champ de l'année dans MongoDB (défaut: "ev_year")
            month_field: Champ du mois dans MongoDB (défaut: "ev_month")

        Returns:
            Liste de dicts [{year, month, count}] triée chronologiquement.
        """
        collection = self._get_collection("accidents")

        pipeline = [
            {"$match": LEAKAGE_GUARD},  # Garde-fou — TOUJOURS en premier
            {
                "$group": {
                    "_id": {
                        "year": f"${year_field}",
                        "month": f"${month_field}",
                    },
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id.year": 1, "_id.month": 1}},
            {
                "$project": {
                    "_id": 0,
                    "year": "$_id.year",
                    "month": "$_id.month",
                    "count": 1,
                }
            },
        ]

        try:
            return list(collection.aggregate(pipeline))
        except OperationFailure as exc:
            logger.error("Échec time series MongoDB : %s", exc)
            raise

    def get_risk_breakdown(self) -> list[dict]:
        """
        Distribution des niveaux de risque (FATL/SERS/MINR/NONE)
        sur les données d'entraînement/validation uniquement.

        Returns:
            Liste de dicts [{_id: "FATL", count: int}, ...]
        """
        # MODIFICATION ICI : On utilise ev_highest_injury au lieu de inj_f_grnd
        return self.get_distribution(group_field="ev_highest_injury")

    def get_stats_summary(self) -> dict[str, Any]:
        """
        Statistiques globales pour le header du dashboard EDA.

        Returns:
            Dict avec total_accidents, years_covered, states_covered.
        """
        collection = self._get_collection("accidents")
        safe_filter = self._safe_filter()

        try:
            total = collection.count_documents(safe_filter)

            years = collection.distinct("ev_year", safe_filter)
            states = collection.distinct("ev_state", safe_filter)

            return {
                "total_accidents": total,
                "years_covered": sorted([y for y in years if y]),
                "states_count": len([s for s in states if s]),
            }
        except OperationFailure as exc:
            logger.error("Échec stats summary MongoDB : %s", exc)
            raise

    # ── Fermeture propre ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Ferme la connexion MongoDB proprement."""
        self._client.close()
        logger.info("Connexion MongoDB fermée.")

    # 👇 LA NOUVELLE FONCTION EST BIEN À L'INTÉRIEUR DE LA CLASSE 👇
    def get_random_example(self) -> dict | None:
        """
        Retourne un document aléatoire depuis la collection accidents
        (hors données test), filtré sur les colonnes RAW_INPUT_COLS.
        """
        from utils.feature_engineering import RAW_INPUT_COLS

        collection = self._get_collection("accidents")

        projection = {col: 1 for col in RAW_INPUT_COLS}
        projection["_id"] = 0

        pipeline = [
            {"$match": LEAKAGE_GUARD},
            {"$sample": {"size": 1}},
            {"$project": projection},
        ]

        try:
            results = list(collection.aggregate(pipeline))
            if not results:
                return None

            doc = results[0]

            cleaned = {
                k: (None if (v != v or v == float("inf") or v == float("-inf")) else v)
                for k, v in doc.items()
                if v is not None and v == v
            }
            return cleaned

        except Exception as exc:
            logger.error("Échec get_random_example MongoDB : %s", exc)
            return None


# ── Instance singleton ────────────────────────────────────────────────────────
# (Attention, ici on est de retour tout à gauche, en dehors de la classe)
_mongo_uri  = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
_db_name    = os.getenv("MONGO_DB_NAME", "aviation_risk")

try:
    mongo_service = MongoService(uri=_mongo_uri, db_name=_db_name)
except Exception as e:
    logger.error("IMPOSSIBLE DE CONNECTER MONGODB : %s", e)
    mongo_service = None  # type: ignore[assignment]