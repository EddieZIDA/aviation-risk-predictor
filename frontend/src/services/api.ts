/**
 * services/api.ts — Client API typé
 * ===================================
 * Couche d'abstraction entre les composants React et le backend Flask.
 * Toutes les fonctions sont typées et gèrent les erreurs réseau.
 *
 * Règle : les composants ne connaissent jamais fetch() directement.
 */

import type {
  PredictionPayload,
  PredictApiResponse,
  ReportPayload,
  ReportApiResponse,
  DistributionResponse,
  StatsResponse,
} from "../types";

// ── Configuration ─────────────────────────────────────────────────────────────

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:5005/api";

// ── Helper fetch générique ────────────────────────────────────────────────────

async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  // On parse le JSON même pour les erreurs HTTP (le backend renvoie toujours du JSON)
  const data: T = await response.json();

  if (!response.ok) {
    // Lance une erreur enrichie avec le message backend si disponible
    const errorData = data as { message?: string; code?: string };
    throw new ApiError(
      errorData.message ?? `Erreur HTTP ${response.status}`,
      response.status,
      errorData.code ?? "HTTP_ERROR"
    );
  }

  return data;
}

// ── Classe d'erreur custom ────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly code: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ── Fonctions API typées ──────────────────────────────────────────────────────

/**
 * POST /api/predict
 * Envoie le payload brut (~128 features) et retourne la prédiction MAPIE.
 *
 * @param payload - Objet JSON avec les features de l'accident
 * @returns PredictApiResponse (succès ou erreur structurée)
 */
export async function predictAccident(
  payload: PredictionPayload
): Promise<PredictApiResponse> {
  return apiFetch<PredictApiResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * POST /api/report
 * Envoie la prédiction + features à Gemini pour générer un rapport de sécurité.
 *
 * @param data - { prediction, uncertainty_interval, accident_features }
 * @returns ReportApiResponse avec le rapport structuré Gemini
 */
export async function generateReport(
  data: ReportPayload
): Promise<ReportApiResponse> {
  return apiFetch<ReportApiResponse>("/report", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

/**
 * GET /api/historical/distribution?field=<field>
 * Retourne la distribution d'un champ pour les graphiques EDA.
 *
 * @param field - Champ MongoDB à agréger (ex: "ev_state", "phase_flt_spec")
 * @returns DistributionResponse avec la liste [{_id, count}]
 */
export async function getHistoricalDistribution(
  field: string
): Promise<DistributionResponse> {
  return apiFetch<DistributionResponse>(
    `/historical/distribution?field=${encodeURIComponent(field)}`
  );
}

/**
 * GET /api/historical/stats
 * Retourne les statistiques globales du dataset (hors test).
 */
export async function getHistoricalStats(): Promise<StatsResponse> {
  return apiFetch<StatsResponse>("/historical/stats");
}

/**
 * GET /api/historical/timeseries
 * Retourne la série temporelle mensuelle des accidents.
 */
export async function getTimeSeries() {
  return apiFetch<{ status: string; data: { year: number; month: number; count: number }[] }>(
    "/historical/timeseries"
  );
}

/**
 * GET /api/historical/risk-breakdown
 * Retourne la distribution FATL/SERS/MINR/NONE.
 */
export async function getRiskBreakdown(): Promise<DistributionResponse> {
  return apiFetch<DistributionResponse>("/historical/risk-breakdown");
}

/**
 * GET /api/historical/random-example
 * Retourne un document aléatoire de MongoDB (hors données test)
 * comme payload brut prêt à soumettre au modèle.
 */
export async function getRandomExample(): Promise<{ status: string; data: Record<string, unknown> }> {
  return apiFetch("/historical/random-example");
}