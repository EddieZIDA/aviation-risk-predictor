import { useState, useCallback } from "react";
import type { PredictionPayload, PredictionResponse, GeminiReport } from "../types";
import { predictAccident, generateReport, ApiError } from "../services/api";
import PredictForm from "../components/PredictForm";
import ResultPanel from "../components/ResultPanel";
import GeminiReportComponent from "../components/GeminiReport";

export default function PredictionPage() {
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [currentPayload, setCurrentPayload] = useState<PredictionPayload | null>(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [reportResult, setReportResult] = useState<GeminiReport | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const handlePredict = useCallback(async (payload: PredictionPayload) => {
    setIsPredicting(true);
    setPredictionError(null);
    setReportResult(null);
    setReportError(null);
    setCurrentPayload(payload);

    try {
      const response = await predictAccident(payload);
      if (response.status === "error") {
        setPredictionError(response.message);
        setPredictionResult(null);
        return;
      }
      setPredictionResult(response);
    } catch (err) {
      setPredictionResult(null);
      setPredictionError("Échec de l'analyse. Vérifiez la connexion au service de calcul.");
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
      if (response.status === "error") {
        setReportError(response.message);
        return;
      }
      setReportResult(response.report);
    } catch (err) {
      setReportError("L'IA n'a pas pu générer le rapport. Vérifiez la clé API.");
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
      <div className="split-column output-zone">
        {!predictionResult && !isPredicting && (
          <div className="empty-state">
            <div className="empty-icon"></div>
            <h3>En attente de données</h3>
            <p>Configurez les paramètres de vol à gauche pour lancer l'évaluation du risque.</p>
          </div>
        )}

        {isPredicting && (
          <div className="loading-state">
            <div className="spinner-large" />
            <p>Le moteur de calcul analyse les variables d'incident...</p>
          </div>
        )}

        {predictionError && (
          <div className="error-card">
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
              <div className="error-card" style={{ marginTop: '20px' }}>
                <p>{reportError}</p>
              </div>
            )}

            {isAnalyzing && (
              <div className="gemini-loading">
                <div className="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
                <p>Génération du rapport d'expertise en cours...</p>
              </div>
            )}

            {reportResult && <GeminiReportComponent report={reportResult} />}
          </div>
        )}
      </div>
    </div>
  );
}