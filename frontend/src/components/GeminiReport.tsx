import { useEffect, useRef } from "react";
import type { GeminiReport as GeminiReportType, RiskLevel } from "../types";
import { RISK_META, SEVERITY_ORDER } from "../types";

interface GeminiReportProps { report: GeminiReportType; }

function isHighSeverity(level: RiskLevel): boolean {
  return SEVERITY_ORDER.indexOf(level) >= SEVERITY_ORDER.indexOf("SERS");
}

function ReportSection({
  icon, title, accent, children,
}: {
  icon: React.ReactNode; title: string; accent?: string; children: React.ReactNode;
}) {
  return (
    <div
      className="report-section"
      style={accent ? { borderColor: accent, backgroundColor: `${accent}10` } : {}}
    >
      <div className="report-section-header">
        <span className="report-section-icon" style={accent ? { color: accent } : {}}>{icon}</span>
        <h3 className="report-section-title" style={accent ? { color: accent } : {}}>{title}</h3>
      </div>
      {children}
    </div>
  );
}

export default function GeminiReport({ report }: GeminiReportProps) {
  const {
    risk_summary, contributing_factors, safety_recommendations,
    worst_case_preparedness, confidence_note, _meta,
  } = report;

  const worstCase          = _meta.worst_case_used;
  const majorityPrediction = _meta.majority_prediction;
  const worstMeta          = RISK_META[worstCase];
  const majorityMeta       = RISK_META[majorityPrediction];
  const highSeverity       = isHighSeverity(worstCase);

  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  return (
    <div className="gemini-report" ref={ref}>

      {/* ── Header marine ── */}
      <div className="report-header">
        <div className="report-header-left">
          <div className="report-header-badge">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2a10 10 0 1 0 10 10"/>
              <path d="M12 6v6l4 2"/>
              <circle cx="18" cy="5" r="3" fill="currentColor" stroke="none"/>
            </svg>
            Gemini AI
          </div>
          <div>
            <h2 className="report-title">Rapport d'Analyse de Sécurité</h2>
            <p className="report-subtitle">
              <span className="report-model">{_meta.model_used}</span>
              &nbsp;·&nbsp;Scénario de précaution :&nbsp;
              <span
                className="report-worst-tag"
                style={{ color: worstMeta.color, borderColor: worstMeta.border, backgroundColor: worstMeta.bg }}
              >
                {worstCase}
              </span>
              {majorityPrediction !== worstCase && (
                <span className="report-majority-note">
                  &nbsp;(prédiction majorit. :&nbsp;
                  <span style={{ color: majorityMeta.color }}>{majorityPrediction}</span>)
                </span>
              )}
            </p>
          </div>
        </div>

        <div
          className="report-severity-chip"
          style={{ color: worstMeta.color, borderColor: worstMeta.border }}
        >
          {highSeverity ? (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
          ) : (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
          )}
          {worstMeta.label}
        </div>
      </div>

      {/* ── Corps ── */}
      <div className="report-body">

        <ReportSection
          icon={<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>}
          title="Synthèse exécutive"
        >
          <p className="report-summary-text">{risk_summary}</p>
        </ReportSection>

        <ReportSection
          icon={<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>}
          title="Facteurs de risque identifiés"
        >
          <ul className="report-factors-list">
            {contributing_factors.map((factor, i) => (
              <li key={i} className="report-factor-item">
                <span className="factor-index">{String(i + 1).padStart(2, "0")}</span>
                <span className="factor-text">{factor}</span>
              </li>
            ))}
          </ul>
        </ReportSection>

        <ReportSection
          icon={<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>}
          title="Recommandations de sécurité"
          accent={highSeverity ? worstMeta.color : undefined}
        >
          {highSeverity && (
            <div className="recommendations-alert" style={{ color: worstMeta.color, borderColor: worstMeta.color }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              </svg>
              Calibrées pour le scénario <strong>{worstCase} — {worstMeta.label}</strong>
            </div>
          )}
          <ol className="report-recommendations-list">
            {safety_recommendations.map((rec, i) => (
              <li key={i} className="report-recommendation-item">
                <span
                  className="rec-number"
                  style={highSeverity ? { color: worstMeta.color, borderColor: worstMeta.border } : {}}
                >
                  {i + 1}
                </span>
                <span className="rec-text">{rec}</span>
              </li>
            ))}
          </ol>
        </ReportSection>

        <ReportSection
          icon={<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>}
          title={`Préparation au scénario ${worstCase}`}
          accent={worstMeta.color}
        >
          <p className="report-worstcase-text">{worst_case_preparedness}</p>
        </ReportSection>

        <ReportSection
          icon={<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>}
          title="Comprendre l'intervalle d'incertitude"
        >
          <p className="report-confidence-text">{confidence_note}</p>
        </ReportSection>

      </div>

      {/* ── Disclaimer ── */}
      <div className="report-disclaimer">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        </svg>
        <span>
          <strong>Rapport généré par intelligence artificielle.</strong>
          &nbsp;Ce document est un outil d'aide à la décision. L'expertise humaine
          qualifiée reste indispensable. Ne pas utiliser comme seule base décisionnelle.
        </span>
      </div>
    </div>
  );
}