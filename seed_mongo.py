#!/usr/bin/env python3
"""
seed_mongo.py — Chargement des données NTSB nettoyées dans MongoDB
===================================================================
Charge data/processed/ntsb_clean_final.csv (sortie du NB01) dans la collection
`accidents` qui alimente l'API Flask (exploration + « charger un dossier type »).

Chaque document reçoit un champ `split` ('train' | 'val' | 'test') calculé avec
EXACTEMENT le même découpage temporel que le NB03 (utils.feature_engineering.split_labels).
Le backend filtre systématiquement `split != 'test'` : les données de test
restent ainsi invisibles depuis l'application.

Usage (depuis la racine du projet) :
    python seed_mongo.py
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from utils.feature_engineering import TARGET_COL, split_labels  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "ntsb_clean_final.csv"
COLLECTION = "accidents"
BATCH_SIZE = 1000


def load_environment() -> tuple[str, str]:
    """Lit MONGO_URI et MONGO_DB_NAME depuis backend/.env (valeurs par défaut sinon)."""
    env_path = PROJECT_ROOT / "backend" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning("%s introuvable, valeurs par défaut utilisées", env_path)
    return (
        os.getenv("MONGO_URI", "mongodb://localhost:27017/"),
        os.getenv("MONGO_DB_NAME", "aviation_risk"),
    )


def load_cleaned_data() -> pd.DataFrame:
    """Charge le CSV nettoyé et ajoute la colonne `split` du NB03."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} introuvable : exécutez d'abord le NB01.")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["split"] = split_labels(df["ev_year"])
    logger.info("%d lignes chargées — répartition : %s",
                len(df), df["split"].value_counts().to_dict())
    return df


def to_documents(df: pd.DataFrame) -> list[dict]:
    """DataFrame → documents MongoDB (NaN → null)."""
    return df.astype(object).where(pd.notnull(df), None).to_dict("records")


def seed_mongodb() -> None:
    mongo_uri, db_name = load_environment()
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    collection = client[db_name][COLLECTION]
    logger.info("Connecté à %s / base '%s'", mongo_uri, db_name)

    df = load_cleaned_data()
    documents = to_documents(df)

    collection.drop()
    for start in range(0, len(documents), BATCH_SIZE):
        collection.insert_many(documents[start:start + BATCH_SIZE], ordered=False)

    # Index utilisés par le filtre anti-fuite et les agrégations du dashboard
    collection.create_index("split")
    collection.create_index(TARGET_COL)

    total = collection.count_documents({})
    n_test = collection.count_documents({"split": "test"})
    logger.info("%d documents insérés (dont %d de test, masqués par l'API)", total, n_test)
    if total != len(df):
        raise RuntimeError(f"{len(df)} documents attendus, {total} présents.")


if __name__ == "__main__":
    try:
        seed_mongodb()
    except Exception as exc:
        logger.error("Échec de la migration : %s", exc)
        sys.exit(1)
