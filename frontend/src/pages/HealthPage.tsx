import { useState, useEffect } from "react";

export default function HealthPage() {
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

useEffect(() => {
    // On cible bien le port 5005 ici aussi
    fetch("http://localhost:5005/api/health")
      .then(res => res.json())
      .then(data => {
        setHealth(data);
        setLoading(false);
      })
      .catch(() => {
        setHealth(null);
        setLoading(false);
      });
  }, []);

  // Composant carte de statut réutilisable
  const StatusCard = ({ title, status, desc, isOk }: { title: string, status: string, desc: string, isOk: boolean }) => (
    <div style={{ 
      padding: '24px', 
      border: `1px solid ${isOk ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`, 
      borderRadius: '12px', 
      background: 'var(--bg-elevated)',
      position: 'relative',
      overflow: 'hidden'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h4 style={{ fontSize: '15px', color: 'var(--text-primary)', fontWeight: 600 }}>{title}</h4>
        <div style={{ 
          width: '12px', height: '12px', borderRadius: '50%', 
          backgroundColor: isOk ? '#10b981' : '#ef4444', 
          boxShadow: `0 0 12px ${isOk ? '#10b981' : '#ef4444'}` 
        }} />
      </div>
      <div style={{ fontSize: '20px', fontWeight: 700, color: isOk ? '#10b981' : '#ef4444', marginBottom: '6px' }}>
        {status}
      </div>
      <div style={{ fontSize: '13px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        {desc}
      </div>
    </div>
  );

  return (
    <div className="predict-form" style={{ padding: '30px', minHeight: '100%' }}>
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ fontSize: '22px', fontWeight: 600, color: 'var(--text-primary)' }}>Supervision du Pipeline ML</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '4px' }}>
          Vérification en temps réel de l'intégrité des modèles, de la base de données et des API tierces.
        </p>
      </div>
      
      {loading ? (
        <div className="loading-state" style={{ border: 'none' }}>
          <div className="spinner-large" />
          <p>Analyse des systèmes en cours...</p>
        </div>
      ) : !health ? (
        <div className="error-card">
          <strong>⚠️ Connexion au Backend Perdue</strong>
          <p>Le serveur Flask (Port 5005) ne répond pas. Veuillez vérifier que votre terminal backend est actif.</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          
          <StatusCard 
            title="Moteur de Prédiction (LightGBM)" 
            status={health.model_loaded ? "En Ligne" : "Hors Ligne"} 
            desc={health.model_loaded ? "Modèle principal chargé avec succès." : "Fichier full_pipeline.pkl introuvable."} 
            isOk={health.model_loaded} 
          />
          
          <StatusCard 
            title="Module d'Incertitude (MAPIE)" 
            status={health.model_loaded ? "Calibré" : "Inactif"} 
            desc="Couverture statistique à 90% active." 
            isOk={health.model_loaded} 
          />
          
          <StatusCard 
            title="Base de Données Historique" 
            status={health.mongodb_config?.uri_set ? "Connectée" : "Déconnectée"} 
            desc={`MongoDB // Collection: ${health.mongodb_config?.collection || "N/A"}`} 
            isOk={health.mongodb_config?.uri_set} 
          />
          
          <StatusCard 
            title="Génération de Rapports (LLM)" 
            status={health.gemini_enabled ? "Prêt" : "Clé API Manquante"} 
            desc="API Google Gemini disponible." 
            isOk={health.gemini_enabled} 
          />

        </div>
      )}
    </div>
  );
}