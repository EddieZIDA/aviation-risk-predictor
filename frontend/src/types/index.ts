/**
 * types/index.ts — Contrats de données TypeScript
 * ================================================
 * Source unique de vérité pour tous les types échangés avec le backend.
 * Chaque interface reflète exactement la structure JSON renvoyée par Flask.
 */

// ── Niveaux de risque ─────────────────────────────────────────────────────────

export type RiskLevel = "FATL" | "SERS" | "MINR" | "NONE";

export const RISK_META: Record<
  RiskLevel,
  { label: string; color: string; bg: string; border: string; glow: string; severity: number }
> = {
  FATL: {
    label: "Fatal",
    color: "#ff2d55",
    bg: "rgba(255,45,85,0.12)",
    border: "rgba(255,45,85,0.45)",
    glow: "0 0 24px rgba(255,45,85,0.4)",
    severity: 4,
  },
  SERS: {
    label: "Grave",
    color: "#ff9500",
    bg: "rgba(255,149,0,0.12)",
    border: "rgba(255,149,0,0.45)",
    glow: "0 0 24px rgba(255,149,0,0.4)",
    severity: 3,
  },
  MINR: {
    label: "Mineur",
    color: "#ffd60a",
    bg: "rgba(255,214,10,0.12)",
    border: "rgba(255,214,10,0.45)",
    glow: "0 0 24px rgba(255,214,10,0.3)",
    severity: 2,
  },
  NONE: {
    label: "Aucun",
    color: "#30d158",
    bg: "rgba(48,209,88,0.12)",
    border: "rgba(48,209,88,0.45)",
    glow: "0 0 24px rgba(48,209,88,0.35)",
    severity: 1,
  },
};

export const SEVERITY_ORDER: RiskLevel[] = ["NONE", "MINR", "SERS", "FATL"];

export function isRiskLevel(value: unknown): value is RiskLevel {
  return typeof value === "string" && (SEVERITY_ORDER as string[]).includes(value);
}

// ── Réponses génériques ───────────────────────────────────────────────────────

/** Toute erreur de l'API a ce format (voir backend/routes/errors.py). */
export interface ApiErrorResponse {
  status: "error";
  code: string;
  message: string;
}

// ── API : Prédiction ──────────────────────────────────────────────────────────

/**
 * Payload brut envoyé à POST /api/predict : une ligne au format du CSV nettoyé.
 * Les valeurs peuvent être du texte : le backend se charge du typage.
 */
export type PredictionPayload = Record<string, string | number | boolean | null>;

/** Réponse succès de POST /api/predict */
export interface PredictionResponse {
  status: "success";
  prediction: RiskLevel;
  /** 1 - alpha de MAPIE (ex. 0.90) ; null si le modèle n'a pas de MAPIE. */
  confidence_level: number | null;
  /** Ensemble de prédiction MAPIE : classes plausibles au niveau de confiance. */
  uncertainty_interval: RiskLevel[];
  probabilities: Partial<Record<RiskLevel, number>>;
  model: { name: string; type: string };
  /** Nombre de variables renseignées parmi celles attendues par le modèle. */
  input_coverage: { provided: number; expected: number };
  _debug?: { raw_prediction_idx: number };
}

// ── API : Rapport Gemini ──────────────────────────────────────────────────────

/** Payload envoyé à POST /api/report */
export interface ReportPayload {
  prediction: RiskLevel;
  uncertainty_interval: RiskLevel[];
  accident_features: PredictionPayload;
}

/** Structure du rapport généré par Gemini (validée côté backend) */
export interface GeminiReport {
  risk_summary: string;
  contributing_factors: string[];
  safety_recommendations: string[];
  worst_case_preparedness: string;
  confidence_note: string;
  _meta: {
    model_used: string;
    majority_prediction: RiskLevel;
    worst_case_used: RiskLevel;
    uncertainty_set: RiskLevel[];
  };
}

/** Réponse succès de POST /api/report */
export interface ReportResponse {
  status: "success";
  report: GeminiReport;
}

// ── API : Données historiques ─────────────────────────────────────────────────

export interface DistributionItem {
  _id: string | number;
  count: number;
}

export interface DistributionResponse {
  status: "success";
  field?: string;
  data: DistributionItem[];
}

export interface SeverityItem {
  _id: string | number;
  total: number;
  NONE: number;
  MINR: number;
  SERS: number;
  FATL: number;
}

export interface SeverityResponse {
  status: "success";
  field: string;
  data: SeverityItem[];
}

export interface StatsResponse {
  status: "success";
  data: {
    total_accidents: number;
    fatal_accidents: number;
    fatal_rate: number;
    years_covered: number[];
    states_count: number;
  };
}

export interface TimeSeriesResponse {
  status: "success";
  data: { year: number; month: number; count: number }[];
}

export interface RandomExampleResponse {
  status: "success";
  data: PredictionPayload;
}

// ── API : Santé ───────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: "online";
  model_loaded: boolean;
  model: { name: string; type: string; confidence_level: number | null } | null;
  mongodb: { connected: boolean; test_split_protected: boolean };
  gemini_enabled: boolean;
  gemini_model: string | null;
}
