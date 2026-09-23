import { useMemo } from "react";
import type { PredictionResponse, RiskLevel } from "../types";
import { RISK_META, SEVERITY_ORDER } from "../types";

interface ResultPanelProps {
  result: PredictionResponse;
  onRequestReport: () => void;
  isReportLoading: boolean;
}

function worstCase(interval: RiskLevel[]): RiskLevel {
  if (!interval.length) return "NONE";
  return [...interval].sort(
    (a, b) => SEVERITY_ORDER.indexOf(b) - SEVERITY_ORDER.indexOf(a)
  )[0];
}

function RiskBadge({ level, size = "md" }: { level: RiskLevel; size?: "sm" | "md" | "lg" }) {
  const meta = RISK_META[level];
  return (
    <span
      className={`risk-badge risk-badge--${size}`}
      style={{ color: meta.color, backgroundColor: meta.bg, borderColor: meta.border }}
    >
      {level}
      {size !== "sm" && <span className="risk-badge-label">{meta.label}</span>}
    </span>
  );
}

function MapieGauge({ interval, prediction }: { interval: RiskLevel[]; prediction: RiskLevel }) {
  const filledLevels = useMemo(() => {
    const set = new Set(interval);
    return SEVERITY_ORDER.filter((l) => set.has(l));
  }, [interval]);

  const worst    = worstCase(interval);
  const worstMeta = RISK_META[worst];

  return (
    <div className="mapie-gauge">
      <div className="mapie-gauge-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M18 20V10M12 20V4M6 20v-6"/>
        </svg>
        Intervalle d'incertitude MAPIE
      </div>

      <div className="mapie-levels">
        {SEVERITY_ORDER.map((level) => {
          const meta = RISK_META[level];
          const isIn = filledLevels.includes(level);
          const isP  = level === prediction;
          return (
            <div
              key={level}
              className={`mapie-level ${isIn ? "active" : "inactive"} ${isP ? "is-prediction" : ""}`}
              style={isIn ? { backgroundColor: meta.bg, borderColor: meta.border, color: meta.color } : {}}
            >
              <span className="mapie-level-code">{level}</span>
              <span className="mapie-level-name">{meta.label}</span>
              {isP && <span className="mapie-prediction-dot" style={{ background: meta.color }} />}
            </div>
          );
        })}
      </div>

      <div className="mapie-summary" style={{ borderColor: worstMeta.border, backgroundColor: worstMeta.bg }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={worstMeta.color} strokeWidth="2.5">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        <span style={{ color: "var(--text-secondary)" }}>
          {filledLevels.length === 1 ? "Seule classe plausible" : "Classes plausibles"}&nbsp;:&nbsp;
          {filledLevels.map((level) => <RiskBadge key={level} level={level} size="sm" />)}
        </span>
      </div>
    </div>
  );
}

/** Probabilités brutes du classifieur, une barre par classe. */
function ProbabilityBars({ probabilities }: { probabilities: PredictionResponse["probabilities"] }) {
  if (!Object.keys(probabilities).length) return null;
  return (
    <div className="proba-list" aria-label="Probabilités par classe">
      {SEVERITY_ORDER.map((level) => {
        const p = probabilities[level] ?? 0;
        return (
          <div className="proba-row" key={level}>
            <span className="proba-code">{level}</span>
            <div className="proba-track">
              <div className="proba-fill" style={{ width: `${p * 100}%`, background: RISK_META[level].color }} />
            </div>
            <span className="proba-value">{Math.round(p * 100)} %</span>
          </div>
        );
      })}
    </div>
  );
}

// En dessous de ce taux de variables renseignées, l'imputation domine la prédiction.
const LOW_COVERAGE = 0.5;

export default function ResultPanel({ result, onRequestReport, isReportLoading }: ResultPanelProps) {
  const { prediction, confidence_level, uncertainty_interval, probabilities, model, input_coverage } = result;
  const meta           = RISK_META[prediction];
  const confidencePct  = confidence_level != null ? Math.round(confidence_level * 100) : null;
  const coverage       = input_coverage.provided / input_coverage.expected;
  const hasInterval    = uncertainty_interval.length > 0;
  const worst          = worstCase(uncertainty_interval);
  const isWorstDiff    = worst !== prediction;

  return (
    <div className="result-panel">
      {/* ── Bandeau titre ── */}
      <div className="result-panel-header">
        <span className="result-panel-title">Résultat du Pipeline ML</span>
        <span className="result-meta">
          <span>{model.name} · {model.type}</span>
          <span>{input_coverage.provided}/{input_coverage.expected} variables renseignées</span>
        </span>
      </div>

      <div className="result-panel-body">
        {/* ── Prédiction principale ── */}
        <div className="result-main" style={{ borderColor: meta.border, backgroundColor: meta.bg }}>
          <div className="result-main-header">
            <span className="result-main-label">Prédiction majoritaire</span>
            {confidencePct != null && (
              <span className="result-confidence" title="Couverture garantie de l'ensemble de prédiction MAPIE">
                Intervalle à {confidencePct} %
              </span>
            )}
          </div>

          <div className="result-risk-display">
            {/* Orbe circulaire */}
            <div
              className="result-risk-orb"
              style={{ borderColor: meta.border, color: meta.color }}
            >
              <div className="result-risk-orb-inner" style={{ backgroundColor: meta.color }} />
            </div>

            <div className="result-risk-info">
              <RiskBadge level={prediction} size="lg" />
              <p className="result-risk-description">
                {prediction === "FATL" && "Accident avec au moins un décès à bord."}
                {prediction === "SERS" && "Blessures graves nécessitant une hospitalisation."}
                {prediction === "MINR" && "Blessures légères, soins ambulatoires uniquement."}
                {prediction === "NONE" && "Aucune blessure physique — incident sans conséquences."}
              </p>
            </div>
          </div>

          {isWorstDiff && (
            <div
              className="result-worstcase-alert"
              style={{
                borderColor: RISK_META[worst].border,
                backgroundColor: RISK_META[worst].bg,
                color: RISK_META[worst].color,
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              Scénario de précaution&nbsp;:&nbsp;
              <strong>{worst} — {RISK_META[worst].label}</strong>
              &nbsp;reste plausible dans l'intervalle.
            </div>
          )}
        </div>

        <ProbabilityBars probabilities={probabilities} />

        {coverage < LOW_COVERAGE && (
          <p className="coverage-warning">
            Seules {input_coverage.provided} variables sur {input_coverage.expected} sont renseignées :
            les autres sont remplacées par des valeurs typiques du jeu d'entraînement, ce qui rend
            cette prédiction peu spécifique au vol analysé.
          </p>
        )}

        {/* ── Jauge MAPIE ── */}
        {hasInterval && <MapieGauge interval={uncertainty_interval} prediction={prediction} />}

        <div className="result-divider" />

        {/* ── Bouton Gemini ── */}
        <div className="result-gemini-section">
          <div className="result-gemini-info">
            <div className="gemini-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2a10 10 0 1 0 10 10"/>
                <path d="M12 6v6l4 2"/>
                <circle cx="18" cy="5" r="3" fill="currentColor" stroke="none"/>
              </svg>
            </div>
            <div>
              <p className="gemini-label">Analyse approfondie Gemini</p>
              <p className="gemini-sublabel">
                Rapport ancré sur le scénario de précaution&nbsp;
                <strong style={{ color: RISK_META[worst].color }}>{worst}</strong>
              </p>
            </div>
          </div>

          <button
            className={`btn-gemini ${isReportLoading ? "loading" : ""}`}
            onClick={onRequestReport}
            disabled={isReportLoading}
            type="button"
          >
            {isReportLoading ? (
              <><span className="spinner" />Gemini analyse…</>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
                </svg>
                Demander l'Analyse
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}