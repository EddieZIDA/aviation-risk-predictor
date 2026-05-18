import { useState, useEffect } from "react";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { getRiskBreakdown } from "../services/api"; 

// Traducteur pour mapper les IDs de la base de données aux labels et couleurs
const RISK_MAPPING: Record<string | number, { label: string, color: string }> = {
  0: { label: "NONE", color: "#10b981" },
  1: { label: "MINEUR", color: "#eab308" },
  2: { label: "SÉRIEUX", color: "#f97316" },
  3: { label: "FATAL", color: "#ef4444" },
  "NONE": { label: "NONE", color: "#10b981" },
  "MINR": { label: "MINEUR", color: "#eab308" },
  "SERS": { label: "SÉRIEUX", color: "#f97316" },
  "FATL": { label: "FATAL", color: "#ef4444" }
};

export default function ExplorePage() {
  const [stats, setStats] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

useEffect(() => {
    getRiskBreakdown()
      .then(response => {
        const rawData = response.data || [];
        
        // 1. On filtre les valeurs nulles, vides ou non définies
        // 2. On transforme les IDs en beaux labels colorés
        const formattedData = rawData
          .filter((item: any) => item._id !== null && item._id !== "null" && item._id !== "")
          .map((item: any) => ({
            ...item,
            name: RISK_MAPPING[item._id]?.label || `Classe ${item._id}`,
            fill: RISK_MAPPING[item._id]?.color || "#94a3b8"
          }));
          
        setStats(formattedData);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || "Erreur de connexion au serveur.");
        setLoading(false);
      });
  }, []);

  return (
    <div className="predict-form" style={{ padding: '30px', display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ fontSize: '22px', fontWeight: 600, color: 'var(--text-primary)' }}>Aperçu de la Base NTSB</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '4px' }}>Analyse de la répartition historique des gravités.</p>
      </div>

      {loading ? (
        <div className="loading-state" style={{ border: 'none' }}>
            <div className="spinner-large" />
            <p>Chargement des statistiques...</p>
        </div>
      ) : error ? (
        <div className="error-card"><strong>⚠️ Erreur</strong><p>{error}</p></div>
      ) : stats.length === 0 ? (
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'center', 
          height: '400px',
          background: 'var(--bg-elevated)',
          borderRadius: '12px',
          border: '1px solid var(--border-subtle)',
          padding: '40px'
        }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚠️</div>
          <h3 style={{ 
            fontSize: '18px', 
            fontWeight: 600, 
            color: 'var(--text-primary)', 
            textAlign: 'center',
            marginBottom: '8px'
          }}>
            Aucune donnée historique disponible
          </h3>
          <p style={{ 
            fontSize: '14px', 
            color: 'var(--text-muted)', 
            textAlign: 'center',
            maxWidth: '300px'
          }}>
            pour la répartition des risques.
          </p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', flex: 1, minHeight: '400px' }}>
          
          <div style={{ background: 'var(--bg-elevated)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ fontSize: '14px', textAlign: 'center', color: 'var(--text-secondary)', marginBottom: '10px' }}>Distribution des Risques</h3>
            <ResponsiveContainer width="100%" height="90%">
              <PieChart>
                <Pie data={stats} dataKey="count" nameKey="name" cx="50%" cy="50%" innerRadius={70} outerRadius={110} paddingAngle={4}>
                  {stats.map((entry, index) => <Cell key={index} fill={entry.fill} stroke="transparent" />)}
                </Pie>
                <Tooltip contentStyle={{ borderRadius: '8px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          
          <div style={{ background: 'var(--bg-elevated)', padding: '20px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ fontSize: '14px', textAlign: 'center', color: 'var(--text-secondary)', marginBottom: '10px' }}>Volume d'Incidents</h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart data={stats} layout="vertical" margin={{ left: 30, right: 30 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#ccc" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={30}>
                   {stats.map((entry, index) => <Cell key={index} fill={entry.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

        </div>
      )}
    </div>
  );
}