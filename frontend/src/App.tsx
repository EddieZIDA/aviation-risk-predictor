import { useState, useEffect, lazy, Suspense } from "react";
import PredictionPage from "./pages/PredictionPage";
import HealthPage from "./pages/HealthPage";
import { getHealth } from "./services/api";
import type { HealthResponse } from "./types";
import "./index.css";

// Recharts pèse ~350 Ko : la page Exploration n'est chargée qu'à la première visite.
const ExplorePage = lazy(() => import("./pages/ExplorePage"));

type Tab = "predict" | "explore" | "health";
type Theme = "light" | "dark";

const TABS: { id: Tab; label: string; title: string }[] = [
  { id: "predict", label: "Prédiction", title: "Analyse Prédictive des Risques d'Accidents Aériens" },
  { id: "explore", label: "Exploration", title: "Statistiques de la Base NTSB" },
  { id: "health", label: "Diagnostic", title: "État des Ressources et Modèles" },
];

/** Onglet lu dans l'URL (#explore...) : permet les favoris et le bouton Précédent. */
function tabFromHash(): Tab {
  const id = window.location.hash.slice(1);
  return TABS.some((t) => t.id === id) ? (id as Tab) : "predict";
}

/** Thème initial : choix mémorisé, sinon préférence du système. */
function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* stockage indisponible (navigation privée...) */
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Badge d'en-tête : résume l'état réel des services. */
function StatusBadge({ health, failed }: { health: HealthResponse | null; failed: boolean }) {
  if (failed) return <div className="user-badge user-badge--down">Backend injoignable</div>;
  if (!health) return <div className="user-badge">Connexion…</div>;
  const degraded = !health.model_loaded || !health.mongodb.connected || !health.gemini_enabled;
  return degraded
    ? <div className="user-badge user-badge--warn">Service dégradé</div>
    : <div className="user-badge">Système fonctionnel</div>;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>(tabFromHash);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthFailed, setHealthFailed] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* ignoré : le thème reste appliqué pour la session */
    }
  }, [theme]);

  useEffect(() => {
    const onHashChange = () => setActiveTab(tabFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealthFailed(true));
  }, []);

  const current = TABS.find((t) => t.id === activeTab)!;

  return (
    <div className="app-root">
      {/* ── BARRE LATÉRALE ── */}
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
              aria-hidden="true"
            >
              <path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.2-1.1.6L3 8l7 4-3.5 3.5-2.5-.5-1.5 1.5 4 1 1 4 1.5-1.5-.5-2.5 3.5-3.5 4 7c.4-.2.7-.6.6-1.1z"/>
            </svg>
          </div>
          <span className="logo-text">AeroRisk</span>
        </div>

        <nav aria-label="Navigation principale">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={activeTab === tab.id ? "active" : ""}
              aria-current={activeTab === tab.id ? "page" : undefined}
              onClick={() => { window.location.hash = tab.id; }}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <button
          type="button"
          className="theme-toggle"
          onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
        >
          {theme === "light" ? "Mode sombre" : "Mode clair"}
        </button>
      </aside>

      {/* ── ZONE DE CONTENU ── */}
      <main className="main-stage">
        <header className="main-header">
          <h1>{current.title}</h1>
          <StatusBadge health={health} failed={healthFailed} />
        </header>

        <div className="content-container">
          {activeTab === "predict" && <PredictionPage />}
          {activeTab === "explore" && (
            <Suspense fallback={<div className="loading-state"><div className="spinner-large" /></div>}>
              <ExplorePage />
            </Suspense>
          )}
          {activeTab === "health" && <HealthPage />}
        </div>
      </main>
    </div>
  );
}
