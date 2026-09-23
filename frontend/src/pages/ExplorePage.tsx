import { useState, useEffect, useMemo } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Cell, Legend, LabelList,
} from "recharts";
import {
  getHistoricalStats, getRiskBreakdown, getSeverityBy, getTimeSeries, errorMessage,
} from "../services/api";
import type { RiskLevel, SeverityItem, StatsResponse } from "../types";
import { RISK_META, SEVERITY_ORDER, isRiskLevel } from "../types";

/**
 * Rampe ordinale à une seule teinte pour la gravité (NONE → FATL).
 * Validée (dataviz/validate_palette.js --ordinal) sur les surfaces
 * --bg-elevated clair (#eef0f5) et sombre (#1f2637) : en sombre, l'ancre
 * s'inverse (le plus grave est le plus lumineux).
 */
const SEVERITY_RAMP: Record<"light" | "dark", Record<RiskLevel, string>> = {
  light: { NONE: "#e5886a", MINR: "#cc573f", SERS: "#a02b29", FATL: "#651317" },
  dark: { NONE: "#9a3b2b", MINR: "#cc5a42", SERS: "#ea8c6c", FATL: "#f8c9b2" },
};

const LINE_COLOR = "#2563eb"; // --accent-blue : série unique

/** Variables proposées pour le croisement avec la gravité (liste blanche backend). */
const CROSS_FIELDS: { value: string; label: string }[] = [
  { value: "light_cond", label: "Conditions lumineuses" },
  { value: "wx_cond_basic", label: "Conditions météo (VMC/IMC)" },
  { value: "acft_category", label: "Catégorie d'aéronef" },
  { value: "type_fly", label: "Type de vol" },
  { value: "far_part", label: "Réglementation FAR" },
  { value: "ev_season", label: "Saison" },
  { value: "num_eng", label: "Nombre de moteurs" },
  { value: "homebuilt", label: "Construction amateur" },
];

/** Libellés français des codes NTSB les plus courants (le code brut reste en infobulle). */
const CODE_LABELS: Record<string, string> = {
  DAYL: "Jour", NITE: "Nuit", DUSK: "Crépuscule", DAWN: "Aube",
  NDRK: "Nuit noire", NBRT: "Nuit éclairée", NR: "Non renseigné",
  VMC: "Vue (VMC)", IMC: "Instruments (IMC)",
  AIR: "Avion", HELI: "Hélicoptère", GLI: "Planeur", BALL: "Ballon", GYRO: "Autogire",
  WSFT: "ULM pendulaire", PPAR: "Paramoteur", PLFT: "Planeur ultra-léger",
  PERS: "Personnel", INST: "Instruction", BUS: "Affaires", AAPL: "Épandage",
  POSI: "Convoyage", OWRK: "Travail aérien", FLTS: "Essai en vol",
  winter: "Hiver", spring: "Printemps", summer: "Été", fall: "Automne",
  Y: "Oui", N: "Non",
};
const codeLabel = (code: string) => CODE_LABELS[code] ?? code;

const nf = new Intl.NumberFormat("fr-FR");
const pct = (v: number) => `${(v * 100).toFixed(1)} %`;

const tooltipStyle = {
  background: "var(--bg-surface)",
  border: "1px solid var(--border-mid)",
  borderRadius: 8,
  color: "var(--text-primary)",
  fontSize: 12.5,
};
const axisTick = { fill: "var(--text-muted)", fontSize: 12 };
const gridStroke = "var(--border-subtle)";

function useTheme(): "light" | "dark" {
  const read = () => (document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  const [theme, setTheme] = useState<"light" | "dark">(read);
  useEffect(() => {
    const obs = new MutationObserver(() => setTheme(read()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);
  return theme;
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="kpi-tile">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
    </div>
  );
}

export default function ExplorePage() {
  const theme = useTheme();
  const ramp = SEVERITY_RAMP[theme];

  const [stats, setStats] = useState<StatsResponse["data"] | null>(null);
  const [breakdown, setBreakdown] = useState<{ level: RiskLevel; count: number; share: number }[]>([]);
  const [yearly, setYearly] = useState<{ year: number; count: number }[]>([]);
  const [crossField, setCrossField] = useState(CROSS_FIELDS[0].value);
  const [severity, setSeverity] = useState<SeverityItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Chargement initial : KPI, distribution de la cible, série annuelle
  useEffect(() => {
    Promise.all([getHistoricalStats(), getRiskBreakdown(), getTimeSeries()])
      .then(([s, b, t]) => {
        setStats(s.data);

        const total = b.data.reduce((acc, d) => acc + d.count, 0);
        setBreakdown(
          SEVERITY_ORDER.map((level) => {
            const count = b.data.find((d) => d._id === level)?.count ?? 0;
            return { level, count, share: total ? count / total : 0 };
          })
        );

        const byYear = new Map<number, number>();
        t.data.forEach((d) => d.year && byYear.set(d.year, (byYear.get(d.year) ?? 0) + d.count));
        // L'année en cours est incomplète : l'afficher ferait croire à une chute.
        const currentYear = new Date().getFullYear();
        setYearly(
          [...byYear]
            .filter(([year]) => year < currentYear)
            .map(([year, count]) => ({ year, count }))
            .sort((a, b) => a.year - b.year)
        );
      })
      .catch((err) => setError(errorMessage(err)))
      .finally(() => setLoading(false));
  }, []);

  // Gravité croisée : rechargée à chaque changement de variable
  useEffect(() => {
    getSeverityBy(crossField)
      .then((res) => setSeverity(res.data))
      .catch((err) => setError(errorMessage(err)));
  }, [crossField]);

  // Parts (%) par modalité pour les barres empilées à 100 %
  const severityShares = useMemo(
    () =>
      severity.slice(0, 10).map((row) => ({
        name: codeLabel(String(row._id)),
        total: row.total,
        ...Object.fromEntries(SEVERITY_ORDER.map((l) => [l, row.total ? row[l] / row.total : 0])),
      })),
    [severity]
  );

  const crossLabel = CROSS_FIELDS.find((f) => f.value === crossField)?.label ?? crossField;

  if (loading) {
    return (
      <div className="page-card">
        <div className="loading-state" style={{ border: "none" }}>
          <div className="spinner-large" />
          <p>Chargement des statistiques</p>
        </div>
      </div>
    );
  }

  if (error && !stats) {
    return (
      <div className="page-card">
        <div className="error-card" role="alert"><strong>Erreur</strong><p>{error}</p></div>
      </div>
    );
  }

  const years = stats?.years_covered ?? [];

  return (
    <div className="page-card">
      <div className="page-intro">
        <h2>Aperçu de la base NTSB</h2>
        <p>Données d'entraînement et de validation uniquement : le jeu de test reste invisible.</p>
      </div>

      {stats && (
        <div className="kpi-grid">
          <Kpi label="Accidents analysés" value={nf.format(stats.total_accidents)} />
          <Kpi label="Accidents mortels" value={pct(stats.fatal_rate)} />
          <Kpi label="Période" value={years.length ? `${years[0]}–${years[years.length - 1]}` : "—"} />
          <Kpi label="États couverts" value={String(stats.states_count)} />
        </div>
      )}

      <div className="chart-grid">
        {/* ── Distribution de la gravité ── */}
        <div className="chart-card">
          <div className="chart-card-header"><h3>Gravité maximale des blessures</h3></div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={breakdown} layout="vertical" margin={{ left: 8, right: 64 }} barCategoryGap={6}>
              <CartesianGrid horizontal={false} stroke={gridStroke} />
              <XAxis type="number" tick={axisTick} tickFormatter={(v) => nf.format(v)} stroke={gridStroke} />
              <YAxis
                type="category" dataKey="level" width={70} tick={axisTick} stroke={gridStroke}
                tickFormatter={(l: string) => (isRiskLevel(l) ? RISK_META[l].label : l)}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ fill: "var(--border-subtle)" }}
                formatter={(v, _n, item) => [`${nf.format(Number(v))} (${pct(Number(item.payload?.share ?? 0))})`, "Accidents"]}
                labelFormatter={(l) => (isRiskLevel(l) ? `${l} — ${RISK_META[l].label}` : l)}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={22} isAnimationActive={false}>
                {breakdown.map((d) => <Cell key={d.level} fill={ramp[d.level]} />)}
                <LabelList
                  dataKey="share" position="right" formatter={(v) => pct(Number(v))}
                  style={{ fill: "var(--text-secondary)", fontSize: 12 }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* ── Série annuelle ── */}
        <div className="chart-card">
          <div className="chart-card-header"><h3>Accidents par année</h3></div>
          <p className="chart-note">Années complètes uniquement</p>
          <ResponsiveContainer width="100%" height={222}>
            <LineChart data={yearly} margin={{ left: 0, right: 16, top: 8 }}>
              <CartesianGrid vertical={false} stroke={gridStroke} />
              <XAxis dataKey="year" tick={axisTick} stroke={gridStroke} />
              <YAxis tick={axisTick} stroke={gridStroke} width={48} tickFormatter={(v) => nf.format(v)} />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(v) => [nf.format(Number(v)), "Accidents"]}
                cursor={{ stroke: "var(--text-faint)" }}
              />
              <Line
                type="linear" dataKey="count" stroke={LINE_COLOR} strokeWidth={2}
                dot={false} activeDot={{ r: 5 }} isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* ── Gravité croisée ── */}
        <div className="chart-card chart-card--wide">
          <div className="chart-card-header">
            <h3>Répartition de la gravité selon : {crossLabel}</h3>
            <label>
              <span className="visually-hidden">Variable croisée</span>
              <select value={crossField} onChange={(e) => setCrossField(e.target.value)}>
                {CROSS_FIELDS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
            </label>
          </div>
          <p className="chart-note">10 modalités les plus fréquentes · part de chaque niveau de gravité (100 % par barre)</p>
          <ResponsiveContainer width="100%" height={Math.max(200, severityShares.length * 34 + 70)}>
            <BarChart data={severityShares} layout="vertical" margin={{ left: 8, right: 16 }} barCategoryGap={6}>
              <CartesianGrid horizontal={false} stroke={gridStroke} />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)} %`} tick={axisTick} stroke={gridStroke} />
              <YAxis type="category" dataKey="name" width={120} tick={axisTick} stroke={gridStroke} />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ fill: "var(--border-subtle)" }}
                formatter={(v, name) => [pct(Number(v)), isRiskLevel(name) ? `${name} — ${RISK_META[name].label}` : String(name)]}
              />
              <Legend
                itemSorter={(item) => SEVERITY_ORDER.indexOf(item.value as RiskLevel)}
                formatter={(value: string) => (
                  <span style={{ color: "var(--text-secondary)", fontSize: 12 }}>
                    {isRiskLevel(value) ? RISK_META[value].label : value}
                  </span>
                )}
              />
              {SEVERITY_ORDER.map((level, i) => (
                <Bar
                  key={level}
                  dataKey={level}
                  stackId="sev"
                  fill={ramp[level]}
                  stroke="var(--bg-elevated)"
                  strokeWidth={2}
                  barSize={22}
                  isAnimationActive={false}
                  radius={i === SEVERITY_ORDER.length - 1 ? [0, 4, 4, 0] : 0}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>

          {/* Équivalent tabulaire (accessibilité, lecture exacte des valeurs) */}
          <details className="table-view">
            <summary>Voir les données</summary>
            <table>
              <thead>
                <tr>
                  <th scope="col">{crossLabel}</th>
                  <th scope="col">Accidents</th>
                  {SEVERITY_ORDER.map((l) => <th scope="col" key={l}>{RISK_META[l].label}</th>)}
                </tr>
              </thead>
              <tbody>
                {severity.map((row) => (
                  <tr key={String(row._id)}>
                    <th scope="row">{codeLabel(String(row._id))} <span style={{ color: "var(--text-faint)" }}>({String(row._id)})</span></th>
                    <td>{nf.format(row.total)}</td>
                    {SEVERITY_ORDER.map((l) => <td key={l}>{pct(row.total ? row[l] / row.total : 0)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </div>
      </div>
    </div>
  );
}
