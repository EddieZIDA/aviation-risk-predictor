import { useState, useEffect } from "react";
import PredictionPage from "./pages/PredictionPage";
import ExplorePage from "./pages/ExplorePage"; // ⬅️ On importe la vraie page
import HealthPage from "./pages/HealthPage";   // ⬅️ On importe la vraie page
import "./index.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "light");

  // Gestion du thème Clair/Sombre
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light');

  return (
    <div className="app-root">
      {/* ── BARRE LATÉRALE (SIDEBAR) ── */}
      <aside className="sidebar-simple">
        <div className="sidebar-logo">
        <div className="logo-icon-wrapper">
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2" 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            className="logo-icon"
          >
            <path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.2-1.1.6L3 8l7 4-3.5 3.5-2.5-.5-1.5 1.5 4 1 1 4 1.5-1.5-.5-2.5 3.5-3.5 4 7c.4-.2.7-.6.6-1.1z"/>
          </svg>
        </div>
        <span className="logo-text">AeroRisk</span>
      </div>
        <nav>
          <button 
            className={activeTab === 'predict' ? 'active' : ''} 
            onClick={() => setActiveTab('predict')}
          >
            Prediction
          </button>
          <button 
            className={activeTab === 'explore' ? 'active' : ''} 
            onClick={() => setActiveTab('explore')}
          >
            Exploration
          </button>
          <button 
            className={activeTab === 'health' ? 'active' : ''} 
            onClick={() => setActiveTab('health')}
          >
            Diagnostic
          </button>
        </nav>

        {/* Bouton de thème en bas du menu */}
        {/* <button 
          onClick={toggleTheme} 
          style={{ marginTop: 'auto', background: 'rgba(255,255,255,0.05)', fontSize: '12px', textAlign: 'center' }}
        >
          {theme === 'light' ? '🌙 Mode Sombre' : '☀️ Mode Clair'}
        </button> */}
      </aside>

      {/* ── ZONE DE CONTENU ── */}
      <main className="main-stage">
        <header className="main-header">
          <h1>
            {activeTab === 'predict' && "Analyse Prédictive en Temps Réel"}
            {activeTab === 'explore' && "Statistiques de la Base NTSB"}
            {activeTab === 'health' && "État des Ressources et Modèles"}
          </h1>
          <div className="user-badge">Système fonctionnel</div>
        </header>

        <div className="content-container">
          {/* ⬅️ Ici, on appelle tes vrais composants ! */}
          {activeTab === 'predict' && <PredictionPage />}
          {activeTab === 'explore' && <ExplorePage />}
          {activeTab === 'health' && <HealthPage />}
        </div>
      </main>
    </div>
  );
}