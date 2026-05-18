import { useState, useCallback, useRef } from "react";
import type { PredictionPayload } from "../types";
import { getRandomExample } from "../services/api";

interface PredictFormProps {
  onSubmit: (payload: PredictionPayload) => Promise<void>;
  isLoading: boolean;
}

export default function PredictForm({ onSubmit, isLoading }: PredictFormProps) {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [dataSourceInfo, setDataSourceInfo] = useState<string | null>(null);
  const [loadingExample, setLoadingExample] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── 1. Charger un exemple depuis la base ──────────────────────────────
  const handleLoadExample = useCallback(async () => {
    setLoadingExample(true);
    setErrorMsg(null);
    try {
      const response = await getRandomExample();
      if (response.status !== "success" || !response.data) {
        setErrorMsg("Connexion à la base de données impossible.");
        return;
      }
      const payload = response.data as Record<string, any>;
      setFormData(payload);
      const make = payload.acft_make || "Aéronef inconnu";
      const year = payload.ev_year || "";
      setDataSourceInfo(`Dossier historique chargé : ${make} ${year}`);
    } catch {
      setErrorMsg("Erreur réseau. Le serveur est-il actif ?");
    } finally {
      setLoadingExample(false);
    }
  }, []);

  // ── 2. Importer un fichier CSV ────────────────────────────────────────
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const text = evt.target?.result as string;
        const lines = text.split('\n').map(l => l.trim()).filter(l => l);
        if (lines.length < 2) throw new Error("Le CSV doit contenir des en-têtes et au moins une ligne.");
        const headers = lines[0].split(',');
        const values = lines[1].split(',');
        const newForm: Record<string, any> = {};
        headers.forEach((h, i) => {
          let val: any = values[i] ? values[i].replace(/^"|"$/g, '') : "";
          if (!isNaN(Number(val)) && val !== "") val = Number(val);
          newForm[h.replace(/^"|"$/g, '')] = val;
        });
        setFormData(newForm);
        setDataSourceInfo(`Fichier importé : ${file.name}`);
        setErrorMsg(null);
      } catch {
        setErrorMsg("Erreur de lecture du fichier CSV. Vérifiez le format.");
      }
    };
    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── 3. Modifier une valeur ────────────────────────────────────────────
  const updateField = (key: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [key]: value === "" ? "" : !isNaN(Number(value)) ? Number(value) : value,
    }));
  };

  const handleClear = useCallback(() => {
    setFormData({});
    setErrorMsg(null);
    setDataSourceInfo(null);
  }, []);

  const handleSubmit = useCallback(async () => {
    setErrorMsg(null);
    if (Object.keys(formData).length === 0) {
      setErrorMsg("Veuillez charger des données avant de lancer l'analyse.");
      return;
    }
    await onSubmit(formData as PredictionPayload);
  }, [formData, onSubmit]);

  const hasData = Object.keys(formData).length > 0;

  return (
    <div className="predict-form" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* ── Header ── */}
      <div style={{
        padding: '18px 22px',
        borderBottom: '1px solid var(--border-light)',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
        background: 'var(--bg-surface)',
      }}>
        {/* Titre */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h2 style={{
              fontFamily: 'var(--font-display)',
              fontSize: '16px',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '-0.01em',
            }}>
              Paramètres de Vol
            </h2>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px', lineHeight: 1.4 }}>
              Saisissez les variables ou importez un jeu de données.
            </p>
          </div>

          {hasData && (
            <button
              type="button"
              onClick={handleClear}
              style={{
                background: 'transparent',
                border: '1px solid rgba(220,38,38,0.25)',
                color: '#dc2626',
                cursor: 'pointer',
                fontSize: '12px',
                fontFamily: 'var(--font-ui)',
                fontWeight: 500,
                padding: '5px 10px',
                borderRadius: '6px',
                transition: 'all 0.15s',
              }}
            >
              Effacer
            </button>
          )}
        </div>

        {/* Boutons d'import — ligne séparée pour éviter l'encombrement */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <input
            type="file"
            accept=".csv"
            ref={fileInputRef}
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            style={{
              flex: 1,
              padding: '8px 12px',
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-mid)',
              borderRadius: '7px',
              cursor: 'pointer',
              fontSize: '12.5px',
              fontFamily: 'var(--font-ui)',
              fontWeight: 500,
              color: 'var(--text-secondary)',
              transition: 'background 0.15s',
            }}
          >
            Importer CSV
          </button>

          <button
            type="button"
            onClick={handleLoadExample}
            disabled={loadingExample || isLoading}
            style={{
              flex: 1,
              padding: '8px 12px',
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-mid)',
              borderRadius: '7px',
              cursor: loadingExample || isLoading ? 'not-allowed' : 'pointer',
              fontSize: '12.5px',
              fontFamily: 'var(--font-ui)',
              fontWeight: 500,
              color: 'var(--text-secondary)',
              opacity: loadingExample || isLoading ? 0.6 : 1,
              transition: 'all 0.15s',
            }}
          >
            {loadingExample ? "Chargement…" : "Charger un dossier type"}
          </button>
        </div>
      </div>

      {/* ── Corps : Grille de données ── */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px 22px',
        background: 'var(--bg-canvas)',
      }}>
        {dataSourceInfo && (
          <div style={{
            padding: '9px 14px',
            background: 'rgba(56,161,105,0.08)',
            border: '1px solid rgba(56,161,105,0.3)',
            borderRadius: '7px',
            color: '#38a169',
            fontSize: '12.5px',
            fontWeight: 500,
            marginBottom: '18px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            ✓ {dataSourceInfo}
          </div>
        )}

        {!hasData ? (
          <div style={{
            textAlign: 'center',
            padding: '60px 20px',
            color: 'var(--text-muted)',
          }}>
            <div style={{ fontSize: '36px', marginBottom: '12px', opacity: 0.35 }}>📂</div>
            <p style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-secondary)' }}>
              Aucune donnée chargée
            </p>
            <p style={{ fontSize: '12.5px', marginTop: '6px', lineHeight: 1.5 }}>
              Utilisez les boutons ci-dessus pour importer un CSV ou charger un dossier de la base.
            </p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px 14px' }}>
            {Object.entries(formData).map(([key, value]) => (
              <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                <label style={{
                  fontSize: '10px',
                  color: 'var(--text-muted)',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.07em',
                }}>
                  {key.replace(/_/g, ' ')}
                </label>
                <input
                  type="text"
                  value={value?.toString() || ""}
                  onChange={(e) => updateField(key, e.target.value)}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Footer : Lancement ── */}
      <div style={{
        padding: '16px 22px',
        borderTop: '1px solid var(--border-light)',
        background: 'var(--bg-surface)',
      }}>
        {errorMsg && (
          <div style={{
            color: '#dc2626',
            fontSize: '12.5px',
            marginBottom: '10px',
            padding: '8px 12px',
            background: 'rgba(220,38,38,0.07)',
            border: '1px solid rgba(220,38,38,0.2)',
            borderRadius: '6px',
          }}>
            {errorMsg}
          </div>
        )}
        <button
          onClick={handleSubmit}
          disabled={!hasData || isLoading}
          style={{
            width: '100%',
            padding: '13px',
            background: hasData && !isLoading ? 'var(--text-primary)' : 'var(--bg-elevated)',
            color: hasData && !isLoading ? 'white' : 'var(--text-muted)',
            border: 'none',
            borderRadius: '9px',
            fontFamily: 'var(--font-display)',
            fontSize: '14px',
            fontWeight: 700,
            letterSpacing: '0.01em',
            cursor: hasData && !isLoading ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s',
            boxShadow: hasData && !isLoading ? '0 2px 8px rgba(0,0,0,0.15)' : 'none',
          }}
        >
          {isLoading ? "Traitement en cours…" : "Lancer l'Évaluation du Risque"}
        </button>
      </div>
    </div>
  );
}