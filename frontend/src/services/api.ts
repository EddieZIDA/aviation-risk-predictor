/**
 * services/api.ts — Client API typé
 * ===================================
 * Couche d'abstraction entre les composants React et le backend Flask.
 * Règle : les composants ne connaissent jamais fetch() directement.
 *
 * Toute réponse non-2xx lève une ApiError portant le message du backend :
 * les composants n'ont qu'à afficher `err.message`.
 */

import type {
  DistributionResponse,
  HealthResponse,
  PredictionPayload,
  PredictionResponse,
  RandomExampleResponse,
  ReportPayload,
  ReportResponse,
  SeverityResponse,
  StatsResponse,
  TimeSeriesResponse,
} from "../types";

// ── Configuration ─────────────────────────────────────────────────────────────

// Par défaut « /api » : en dev, le proxy Vite redirige vers Flask (vite.config.ts),
// ce qui évite CORS. VITE_API_URL permet de pointer vers un autre serveur.
const BASE_URL = import.meta.env.VITE_API_URL ?? "/api";

// ── Erreur typée ──────────────────────────────────────────────────────────────

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

/** Message lisible pour n'importe quelle erreur attrapée dans un composant. */
export function errorMessage(err: unknown, fallback = "Une erreur inattendue s'est produite."): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof TypeError) return "Serveur injoignable. Le backend Flask est-il démarré ?";
  return fallback;
}

// ── Helper fetch générique ────────────────────────────────────────────────────

async function apiFetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });

  // Le backend répond toujours en JSON ; un proxy en erreur peut renvoyer du HTML.
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    if (response.ok) throw new ApiError("Réponse invalide du serveur.", response.status, "INVALID_JSON");
  }

  if (!response.ok) {
    const err = (body ?? {}) as { message?: string; code?: string };
    const message =
      err.message ??
      (response.status >= 500
        ? "Le backend ne répond pas correctement (vérifiez qu'il est démarré sur le port 5005)."
        : `Erreur HTTP ${response.status}`);
    throw new ApiError(message, response.status, err.code ?? "HTTP_ERROR");
  }

  return body as T;
}

// ── Prédiction & rapport ──────────────────────────────────────────────────────

/** POST /api/predict — prédiction + ensemble d'incertitude MAPIE. */
export function predictAccident(payload: PredictionPayload): Promise<PredictionResponse> {
  return apiFetch("/predict", { method: "POST", body: JSON.stringify(payload) });
}

/** POST /api/report — rapport de sécurité Gemini ancré sur le pire scénario plausible. */
export function generateReport(data: ReportPayload): Promise<ReportResponse> {
  return apiFetch("/report", { method: "POST", body: JSON.stringify(data) });
}

// ── Données historiques (hors jeu de test) ────────────────────────────────────

export function getHistoricalStats(): Promise<StatsResponse> {
  return apiFetch("/historical/stats");
}

export function getHistoricalDistribution(field: string): Promise<DistributionResponse> {
  return apiFetch(`/historical/distribution?field=${encodeURIComponent(field)}`);
}

export function getSeverityBy(field: string): Promise<SeverityResponse> {
  return apiFetch(`/historical/severity?field=${encodeURIComponent(field)}`);
}

export function getTimeSeries(): Promise<TimeSeriesResponse> {
  return apiFetch("/historical/timeseries");
}

export function getRiskBreakdown(): Promise<DistributionResponse> {
  return apiFetch("/historical/risk-breakdown");
}

/** Document aléatoire de MongoDB, prêt à être soumis à /api/predict. */
export function getRandomExample(): Promise<RandomExampleResponse> {
  return apiFetch("/historical/random-example");
}

// ── Diagnostic ────────────────────────────────────────────────────────────────

export function getHealth(): Promise<HealthResponse> {
  return apiFetch("/health");
}
