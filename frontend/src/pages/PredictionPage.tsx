import { useState, useCallback } from "react";
import type { PredictionPayload, PredictionResponse, GeminiReport } from "../types";
import { predictAccident, generateReport, errorMessage } from "../services/api";
import PredictForm from "../components/PredictForm";
import ResultPanel from "../components/ResultPanel";
import GeminiReportComponent from "../components/GeminiReport";

export default function PredictionPage() {
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  // Payload ayant produit la prédiction affichée (réutilisé pour le rapport Gemini)
  const [currentPayload, setCurrentPayload] = useState<PredictionPayload | null>(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [reportResult, setReportResult] = useState<GeminiReport | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const handlePredict = useCallback(async (payload: PredictionPayload) => {
    setIsPredicting(true);
    setPredictionError(null);
    setPredictionResult(null);
    setReportResult(null);
    setReportError(null);
    setCurrentPayload(payload);

    try {
      setPredictionResult(await predictAccident(payload));
    } catch (err) {
      setPredictionError(errorMessage(err, "Échec de l'analyse."));
    } finally {
      setIsPredicting(false);
    }
  }, []);

  const handleRequestReport = useCallback(async () => {
    if (!predictionResult || !currentPayload) return;
    setIsAnalyzing(true);
    setReportError(null);

    try {
      const response = await generateReport({
        prediction: predictionResult.prediction,
        uncertainty_interval: predictionResult.uncertainty_interval,
        accident_features: currentPayload,
      });
      setReportResult(response.report);
    } catch (err) {
      setReportError(errorMessage(err, "Le rapport n'a pas pu être généré."));
    } finally {
      setIsAnalyzing(false);
    }
  }, [predictionResult, currentPayload]);

  return (
    <div className="split-layout">
      {/* ── BLOC GAUCHE : SAISIE (Fixe) ── */}
      <div className="split-column input-zone">
        <PredictForm onSubmit={handlePredict} isLoading={isPredicting} />
      </div>

      {/* ── BLOC DROIT : RÉSULTATS (Défilant) ── */}
      <div className="split-column output-zone" aria-live="polite">
        {!predictionResult && !isPredicting && !predictionError && (
          <div className="empty-state">
            <div className="empty-icon"></div>
            <h3>En attente de données</h3>
            <p>Configurez les paramètres de vol à gauche pour lancer l'évaluation du risque.</p>
          </div>
        )}

        {isPredicting && (
          <div className="loading-state">
            <div className="spinner-large" />
            <p>Le moteur de calcul analyse les variables d'incident</p>
          </div>
        )}

        {predictionError && (
          <div className="error-card" role="alert">
            <strong>Erreur d'Analyse</strong>
            <p>{predictionError}</p>
          </div>
        )}

        {predictionResult && (
          <div className="analysis-flow">
            <ResultPanel
              result={predictionResult}
              onRequestReport={handleRequestReport}
              isReportLoading={isAnalyzing}
            />

            {reportError && (
              <div className="error-card" role="alert" style={{ marginTop: '20px' }}>
                <strong>Rapport indisponible</strong>
                <p>{reportError}</p>
              </div>
            )}

            {isAnalyzing && (
              <div className="gemini-loading">
                <div className="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
                <p>Génération du rapport d'expertise en cours</p>
              </div>
            )}

            {reportResult && <GeminiReportComponent report={reportResult} />}
          </div>
        )}
      </div>
    </div>
  );
}
