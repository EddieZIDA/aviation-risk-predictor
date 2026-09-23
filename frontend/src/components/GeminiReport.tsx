import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { GeminiReport as GeminiReportType, RiskLevel } from "../types";
import { RISK_META, SEVERITY_ORDER } from "../types";

interface GeminiReportProps { report: GeminiReportType; }

/**
 * Retire une éventuelle puce ajoutée par le LLM (« 1. », « - », « * »...)
 * sans toucher aux phrases qui commencent par un nombre (« 100 heures... »).
 */
function stripListMarker(text: string): string {
  return text.replace(/^\s*(?:[-*•]|\d{1,2}[.)])\s+/, "");
}

function isHighSeverity(level: RiskLevel): boolean {
  return SEVERITY_ORDER.indexOf(level) >= SEVERITY_ORDER.indexOf("SERS");
}

/** Pastille de niveau : point coloré + code + libellé (le texte reste en encre neutre). */
function LevelTag({ level }: { level: RiskLevel }) {
  return (
    <span className="level-tag" style={{ "--level-color": RISK_META[level].color } as CSSProperties}>
      <span className="level-tag-dot" aria-hidden="true" />
      <strong>{level}</strong>
      <span className="level-tag-label">{RISK_META[level].label}</span>
    </span>
  );
}

function Section({ icon, title, className = "", children }: {
  icon: ReactNode; title: string; className?: string; children: ReactNode;
}) {
  return (
    <section className={`report-section ${className}`}>
      <h3 className="report-section-title">
        <span className="report-section-icon" aria-hidden="true">{icon}</span>
        {title}
      </h3>
      {children}
    </section>
  );
}

const Icon = {
  summary: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>,
  factors: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>,
  shield: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>,
  bolt: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
  chart: <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>,
  spark: <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.6L22 12l-7.6 2.4L12 22l-2.4-7.6L2 12l7.6-2.4z"/></svg>,
};

/** Version texte du rapport, pour le presse-papiers. */
function reportToText(report: GeminiReportType): string {
  const { _meta: meta } = report;
  const list = (items: string[]) => items.map((t, i) => `  ${i + 1}. ${stripListMarker(t)}`).join("\n");
  return [
    "RAPPORT D'ANALYSE DE SÉCURITÉ",
    `Classe la plus probable : ${meta.majority_prediction} · Classes plausibles : ${meta.uncertainty_set.join(", ") || "aucune"}`,
    "",
    "Synthèse", report.risk_summary, "",
    "Facteurs de risque", list(report.contributing_factors), "",
    "Recommandations", list(report.safety_recommendations), "",
    meta.worst_case_used === "NONE" ? "Points de vigilance" : `Préparation au scénario ${meta.worst_case_used}`,
    report.worst_case_preparedness, "",
    "Comprendre l'incertitude", report.confidence_note, "",
    `Généré par ${meta.model_used} — aide à la décision, ne remplace pas l'expertise humaine.`,
  ].join("\n");
}

export default function GeminiReport({ report }: GeminiReportProps) {
  const {
    risk_summary, contributing_factors, safety_recommendations,
    worst_case_preparedness, confidence_note, _meta,
  } = report;

  const worst = _meta.worst_case_used;
  const worstMeta = RISK_META[worst];
  const high = isHighSeverity(worst);
  const preparednessTitle = worst === "NONE"
    ? "Points de vigilance"
    : `Préparation au scénario ${worstMeta.label.toLowerCase()} (${worst})`;

  const [copied, setCopied] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(reportToText(report));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* presse-papiers indisponible (contexte non sécurisé) */
    }
  };

  return (
    <article
      className="gemini-report"
      ref={ref}
      style={{ "--report-accent": worstMeta.color } as CSSProperties}
    >
      {/* ── En-tête ── */}
      <header className="report-header">
        <div>
          <p className="report-eyebrow">{Icon.spark} Rapport généré par IA</p>
          <h2 className="report-title">Analyse de sécurité du vol</h2>
          <p className="report-subtitle">{_meta.model_used}</p>
        </div>
        <div className="report-reference">
          <span className="report-reference-label">Scénario de référence</span>
          <LevelTag level={worst} />
        </div>
      </header>

      {/* ── Rappel de l'évaluation ── */}
      <div className="report-context">
        <div>
          <span className="report-context-label">Classe la plus probable</span>
          <LevelTag level={_meta.majority_prediction} />
        </div>
        <div>
          <span className="report-context-label">Classes plausibles à 90 %</span>
          <span className="report-context-tags">
            {_meta.uncertainty_set.length
              ? SEVERITY_ORDER.filter((l) => _meta.uncertainty_set.includes(l)).map((l) => <LevelTag key={l} level={l} />)
              : <span className="report-context-empty">aucune (situation atypique)</span>}
          </span>
        </div>
      </div>

      <div className="report-body">
        <Section icon={Icon.summary} title="Synthèse" className="report-section--lead">
          <p className="report-lead">{risk_summary}</p>
        </Section>

        <div className="report-columns">
          <Section icon={Icon.factors} title="Facteurs de risque">
            <ol className="report-list">
              {contributing_factors.map((factor, i) => (
                <li key={i}>
                  <span className="report-list-index">{String(i + 1).padStart(2, "0")}</span>
                  <span>{stripListMarker(factor)}</span>
                </li>
              ))}
            </ol>
          </Section>

          <Section icon={Icon.shield} title="Recommandations" className={high ? "report-section--accent" : ""}>
            <ol className="report-list report-list--actions">
              {safety_recommendations.map((rec, i) => (
                <li key={i}>
                  <span className="report-list-index">{i + 1}</span>
                  <span>{stripListMarker(rec)}</span>
                </li>
              ))}
            </ol>
          </Section>
        </div>

        <Section icon={Icon.bolt} title={preparednessTitle} className="report-section--accent">
          <p className="report-text">{worst_case_preparedness}</p>
        </Section>

        <Section icon={Icon.chart} title="Comprendre l'incertitude" className="report-section--muted">
          <p className="report-text">{confidence_note}</p>
        </Section>
      </div>

      {/* ── Pied ── */}
      <footer className="report-footer">
        <p className="report-disclaimer">
          <span aria-hidden="true">{Icon.shield}</span>
          <span>
            <strong>Aide à la décision générée par IA.</strong> L'expertise d'un professionnel
            qualifié reste indispensable ; ne pas utiliser comme seule base décisionnelle.
          </span>
        </p>
        <button type="button" className="btn-secondary" onClick={copy}>
          {copied ? "Copié ✓" : "Copier le rapport"}
        </button>
      </footer>
    </article>
  );
}
