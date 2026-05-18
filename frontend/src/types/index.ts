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

// ── API : Prédiction ──────────────────────────────────────────────────────────

/** Payload brut envoyé à POST /api/predict */
export type PredictionPayload = Record<string, string | number | null>;

/** Réponse succès de POST /api/predict */
export interface PredictionResponse {
  status: "success";
  prediction: RiskLevel;
  confidence_level: number;           // ex: 0.90
  uncertainty_interval: RiskLevel[];  // ex: ["MINR", "SERS"]
  _debug?: {
    raw_prediction_idx: number;
  };
}

/** Réponse erreur générique de l'API */
export interface ApiErrorResponse {
  status: "error";
  code: string;
  message: string;
}

export type PredictApiResponse = PredictionResponse | ApiErrorResponse;

// ── API : Rapport Gemini ──────────────────────────────────────────────────────

/** Payload envoyé à POST /api/report */
export interface ReportPayload {
  prediction: RiskLevel;
  uncertainty_interval: RiskLevel[];
  accident_features: PredictionPayload;
}

/** Structure du rapport généré par Gemini (champs JSON stricts) */
export interface GeminiReport {
  report(report: any): unknown;
  message(message: any): unknown;
  status: string;
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

export type ReportApiResponse = ReportResponse | ApiErrorResponse;

// ── API : Données historiques ─────────────────────────────────────────────────

export interface DistributionItem {
  _id: string;
  count: number;
}

export interface DistributionResponse {
  status: "success";
  field: string;
  data: DistributionItem[];
}

export interface StatsResponse {
  status: "success";
  data: {
    total_accidents: number;
    years_covered: number[];
    states_count: number;
  };
}

// ── États UI locaux ───────────────────────────────────────────────────────────

export type AsyncStatus = "idle" | "loading" | "success" | "error";

export interface PredictionState {
  status: AsyncStatus;
  result: PredictionResponse | null;
  error: string | null;
}

export interface ReportState {
  status: AsyncStatus;
  result: GeminiReport | null;
  error: string | null;
}