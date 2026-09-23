import { useState, useEffect } from "react";
import { getHealth, errorMessage } from "../services/api";
import type { HealthResponse } from "../types";

function StatusCard({ title, status, desc, isOk }: { title: string; status: string; desc: string; isOk: boolean }) {
  return (
    <div className={`status-card ${isOk ? "status-card--ok" : "status-card--ko"}`}>
      <div className="status-card-head">
        <h4>{title}</h4>
        <span className="status-dot" aria-hidden="true" />
      </div>
      <div className="status-value">{status}</div>
      <div className="status-desc">{desc}</div>
    </div>
  );
}

export default function HealthPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err) => setError(errorMessage(err)))
      .finally(() => setLoading(false));
  }, []);

  const model = health?.model;
  const confidencePct = model?.confidence_level != null ? Math.round(model.confidence_level * 100) : null;

  return (
    <div className="page-card">
      <div className="page-intro">
        <h2>Supervision du Pipeline ML</h2>
        <p>État en temps réel des modèles, de la base de données et des API tierces.</p>
      </div>

      {loading ? (
        <div className="loading-state" style={{ border: "none" }}>
          <div className="spinner-large" />
          <p>Analyse des systèmes en cours…</p>
        </div>
      ) : !health ? (
        <div className="error-card" role="alert">
          <strong>Connexion au backend impossible</strong>
          <p>{error}</p>
        </div>
      ) : (
        <div className="health-grid">
          <StatusCard
            title="Moteur de prédiction"
            status={health.model_loaded ? "En ligne" : "Hors ligne"}
            desc={model ? `${model.name} chargé` : "Artefacts des notebooks 03 à 05 introuvables."}
            isOk={health.model_loaded}
          />

          <StatusCard
            title="Module d'incertitude"
            status={confidencePct != null ? "Calibré" : "Inactif"}
            desc={confidencePct != null
              ? `${model?.type} · couverture cible ${confidencePct} %`
              : "Pas de MAPIE : prédiction ponctuelle uniquement."}
            isOk={confidencePct != null}
          />

          <StatusCard
            title="Base de données historique"
            status={health.mongodb.connected ? "Connectée" : "Déconnectée"}
            desc={!health.mongodb.connected
              ? "MongoDB injoignable (MONGO_URI)."
              : health.mongodb.test_split_protected
                ? "Collection accidents · jeu de test masqué"
                : "Champ split absent : relancez seed_mongo.py"}
            isOk={health.mongodb.connected && health.mongodb.test_split_protected}
          />

          <StatusCard
            title="Génération de rapports (LLM)"
            status={health.gemini_enabled ? "Prêt" : "Clé API manquante"}
            desc={health.gemini_enabled ? `Google ${health.gemini_model}` : "Définir GEMINI_API_KEY dans backend/.env."}
            isOk={health.gemini_enabled}
          />
        </div>
      )}
    </div>
  );
}
