#!/usr/bin/env python3
"""
Script de migration des données NTSB nettoyées vers MongoDB
===============================================

Ce script charge les données nettoyées depuis le CSV généré par les notebooks
et les insère dans une collection MongoDB pour alimenter l'API Flask.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment():
    """Charge les variables d'environnement depuis backend/.env"""
    # Chemin vers le fichier .env du backend
    env_path = Path(__file__).parent / "backend" / ".env"
    
    # Valeurs par défaut
    defaults = {
        'MONGO_URI': 'mongodb://localhost:27017/',
        'MONGO_DB_NAME': 'aviation_risk'
    }
    
    # Charger le fichier .env s'il existe
    if env_path.exists():
        logger.info(f"Chargement des variables d'environnement depuis {env_path}")
        load_dotenv(env_path)
    else:
        logger.warning(f"Fichier .env non trouvé à {env_path}, utilisation des valeurs par défaut")
    
    # Récupérer les valeurs avec les défauts
    mongo_uri = os.getenv('MONGO_URI', defaults['MONGO_URI'])
    mongo_db_name = os.getenv('MONGO_DB_NAME', defaults['MONGO_DB_NAME'])
    
    return mongo_uri, mongo_db_name

def connect_to_mongodb(uri, db_name):
    """Établit la connexion à MongoDB"""
    try:
        client = MongoClient(uri)
        # Test de connexion
        client.admin.command('ping')
        logger.info(f"Connecté à MongoDB avec succès: {uri}")
        return client[db_name]
    except Exception as e:
        logger.error(f"Erreur de connexion à MongoDB: {e}")
        raise

def load_cleaned_data():
    """Charge les données nettoyées depuis le CSV"""
    csv_path = Path(__file__).parent / "data" / "processed" / "ntsb_clean_final.csv"
    
    if not csv_path.exists():
        logger.error(f"Fichier CSV non trouvé: {csv_path}")
        raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas")
    
    logger.info(f"Chargement des données depuis {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Chargé {len(df)} enregistrements depuis le CSV")
    
    return df

def prepare_data_for_mongodb(df):
    """Prépare les données pour l'insertion dans MongoDB"""
    # Convertir les NaN en None (null pour MongoDB)
    df_clean = df.where(pd.notnull(df), None)
    
    # Convertir le DataFrame en liste de dictionnaires
    records = df_clean.to_dict('records')
    
    logger.info(f"Données préparées: {len(records)} enregistrements")
    return records

def seed_mongodb():
    """Fonction principale de migration"""
    logger.info("=== DÉBUT DE LA MIGRATION MONGODB ===")
    
    try:
        # 1. Charger l'environnement
        mongo_uri, mongo_db_name = load_environment()
        logger.info(f"Base de données cible: {mongo_db_name}")
        
        # 2. Connexion à MongoDB
        db = connect_to_mongodb(mongo_uri, mongo_db_name)
        collection = db.accidents
        
        # 3. Vider la collection existante
        logger.info("Vidage de la collection 'accidents'...")
        result = collection.drop()
        logger.info("Collection vidée avec succès")
        
        # 4. Charger les données nettoyées
        df = load_cleaned_data()
        
        # 5. Préparer les données pour MongoDB
        records = prepare_data_for_mongodb(df)
        
        # 6. Insérer les données
        logger.info(f"Insertion de {len(records)} documents dans MongoDB...")
        
        # Insertion par lots pour éviter les problèmes de mémoire
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                result = collection.insert_many(batch, ordered=False)
                batch_count = len(result.inserted_ids)
                total_inserted += batch_count
                logger.info(f"Lot {i//batch_size + 1}: {batch_count} documents insérés")
            except Exception as e:
                logger.error(f"Erreur lors de l'insertion du lot {i//batch_size + 1}: {e}")
                # En cas d'erreur, essayer d'insérer un par un
                for doc in batch:
                    try:
                        collection.insert_one(doc)
                        total_inserted += 1
                    except Exception as doc_error:
                        logger.error(f"Erreur lors de l'insertion d'un document: {doc_error}")
        
        # 7. Créer l'index sur le champ 'split'
        logger.info("Création de l'index sur le champ 'split'...")
        try:
            collection.create_index("split")
            logger.info("Index sur 'split' créé avec succès")
        except Exception as e:
            logger.warning(f"Erreur lors de la création de l'index: {e}")
        
        # 8. Vérification finale
        total_count = collection.count_documents({})
        logger.info(f"=== RAPPORT FINAL ===")
        logger.info(f"Documents attendus: {len(df)}")
        logger.info(f"Documents insérés: {total_inserted}")
        logger.info(f"Documents dans la collection: {total_count}")
        logger.info(f"Taux de succès: {(total_inserted/len(df)*100):.2f}%")
        
        if total_count == len(df):
            logger.info("✅ MIGRATION TERMINÉE AVEC SUCCÈS")
        else:
            logger.warning("⚠️  MIGRATION TERMINÉE AVEC AVERTISSEMENTS")
            
    except Exception as e:
        logger.error(f"Erreur critique lors de la migration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    seed_mongodb()
