"""
services/gemini_service.py — Service de Génération de Rapports LLM
====================================================================
Responsabilité UNIQUE : construire le prompt de sécurité aéronautique
et appeler l'API Google Gemini. Aucune logique de routing ici.

Principe directeur du prompt :
  Gemini doit toujours raisonner à partir du SCÉNARIO LE PLUS GRAVE
  présent dans l'uncertainty_set MAPIE, pas seulement de la prédiction
  majoritaire. C'est la garantie d'une recommandation conservatrice.
"""

import os
import logging
from typing import Any

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ── Descriptions humaines des niveaux de risque ──────────────────────────────
RISK_DESCRIPTIONS: dict[str, str] = {
    "NONE": "Aucun blessé (incident sans conséquences physiques)",
    "MINR": "Blessures mineures (soins ambulatoires, pas d'hospitalisation)",
    "SERS": "Blessures graves (hospitalisation, séquelles possibles)",
    "FATL": "Accident fatal (décès d'au moins une personne à bord)",
}

# Ordre de gravité croissant — utilisé pour identifier le scénario le pire
SEVERITY_ORDER: list[str] = ["NONE", "MINR", "SERS", "FATL"]


class GeminiService:
    """
    Service singleton pour la génération de rapports de sécurité
    aéronautique via l'API Google Gemini.
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        """
        Configure le client Gemini.

        Args:
            api_key   : Clé API Google AI Studio.
            model_name: Identifiant du modèle (ex: 'gemini-1.5-flash').

        Raises:
            ValueError : Si la clé API est absente.
            RuntimeError: Si la configuration Gemini échoue.
        """
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY est manquante. "
                "Vérifiez votre fichier .env."
            )
        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name=model_name)
            self._model_name = model_name
            logger.info("Gemini initialisé avec le modèle '%s'.", model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Échec de l'initialisation Gemini : {exc}"
            ) from exc

    # ── Identification du scénario le plus grave ─────────────────────────────

    @staticmethod
    def _worst_case_scenario(uncertainty_set: list[str]) -> str:
        """
        Retourne le niveau de risque le plus grave parmi les classes
        plausibles renvoyées par MAPIE.

        Args:
            uncertainty_set: Liste de labels MAPIE (ex: ["MINR", "SERS"])

        Returns:
            Le label de gravité maximale (ex: "SERS")
        """
        if not uncertainty_set:
            return "FATL"  # Fallback conservateur si le set est vide

        sorted_by_severity = sorted(
            uncertainty_set,
            key=lambda label: SEVERITY_ORDER.index(label)
            if label in SEVERITY_ORDER else -1,
            reverse=True,  # Le plus grave en premier
        )
        return sorted_by_severity[0]

    # ── Construction du prompt système ───────────────────────────────────────

    def _build_prompt(
        self,
        prediction: str,
        uncertainty_set: list[str],
        accident_features: dict[str, Any],
    ) -> str:
        """
        Construit le prompt complet envoyé à Gemini.

        Logique clé : le rapport est ancré sur le worst-case scenario
        (scénario le plus grave de l'uncertainty_set), pas sur la
        prédiction majoritaire seule.

        Args:
            prediction       : Classe majoritaire MAPIE (ex: "MINR")
            uncertainty_set  : Toutes les classes plausibles (ex: ["MINR","SERS"])
            accident_features: Dict des features brutes du vol analysé

        Returns:
            Prompt complet en string.
        """
        worst_case = self._worst_case_scenario(uncertainty_set)

        # Formatage lisible des features clés pour le contexte
        key_features = self._format_key_features(accident_features)

        # Formatage de l'intervalle d'incertitude pour le prompt
        uncertainty_str = ", ".join(
            f"{label} ({RISK_DESCRIPTIONS.get(label, label)})"
            for label in uncertainty_set
        ) if uncertainty_set else "Non disponible"

        prompt = f"""Tu es AERO-ANALYST, un expert en sécurité aéronautique de niveau FAA/OACI.
Tu analyses des données d'accidents issues de la base NTSB et génères des rapports
de sécurité professionnels, factuels et exploitables par des pilotes ou enquêteurs.

══════════════════════════════════════════
DONNÉES DE L'ÉVÉNEMENT ANALYSÉ
══════════════════════════════════════════
{key_features}

══════════════════════════════════════════
ÉVALUATION DU RISQUE (Modèle ML + MAPIE)
══════════════════════════════════════════
• Prédiction majoritaire  : {prediction} — {RISK_DESCRIPTIONS.get(prediction, prediction)}
• Intervalle d'incertitude (confiance 90%) :
  {uncertainty_str}
• Scénario de référence pour ce rapport : {worst_case} — {RISK_DESCRIPTIONS.get(worst_case, worst_case)}

══════════════════════════════════════════
INSTRUCTION DE RAISONNEMENT (CRITIQUE)
══════════════════════════════════════════
Le modèle ML prédit "{prediction}" comme classe la plus probable,
MAIS l'intervalle de confiance MAPIE indique que le scénario "{worst_case}"
reste statistiquement plausible à 90% de confiance.

Tu DOIS rédiger tes recommandations de sécurité en te préparant au scénario
"{worst_case}", car c'est le principe de précaution fondamental en aviation.
Ne minimise JAMAIS les risques vers le bas de l'intervalle.

══════════════════════════════════════════
FORMAT DE RÉPONSE ATTENDU (JSON strict)
══════════════════════════════════════════
Réponds UNIQUEMENT avec un objet JSON valide, sans balises Markdown, sans texte avant/après.

{{
  "risk_summary": "Synthèse exécutive en 2-3 phrases. Mentionner explicitement la prédiction majoritaire ET le scénario de précaution.",
  "contributing_factors": [
    "Facteur de risque 1 identifié dans les données",
    "Facteur de risque 2",
    "Facteur de risque 3 (max 5 facteurs)"
  ],
  "safety_recommendations": [
    "Recommandation opérationnelle 1 (concrète, actionnable)",
    "Recommandation opérationnelle 2",
    "Recommandation opérationnelle 3 (max 5 recommandations)"
  ],
  "worst_case_preparedness": "Paragraph décrivant spécifiquement comment se préparer au scénario '{worst_case}', même si la probabilité majoritaire est '{prediction}'.",
  "confidence_note": "Explication pédagogique de l'intervalle MAPIE pour un pilote non-statisticien."
}}"""

        return prompt

    # ── Formatage des features clés ──────────────────────────────────────────

    @staticmethod
    def _format_key_features(features: dict[str, Any]) -> str:
        """
        Extrait et formate un sous-ensemble lisible des features pour
        enrichir le contexte du prompt sans le surcharger.

        Args:
            features: Dict brut des features de l'accident.

        Returns:
            String formaté multi-lignes.
        """
        # Features prioritaires pour le contexte LLM
        priority_keys = [
            ("ev_state",        "État (USA)"),
            ("ev_type",         "Type d'événement"),
            ("ev_year",         "Année"),
            ("ev_month",        "Mois"),
            ("light_cond",      "Conditions lumineuses"),
            ("weather",         "Météo générale"),
            ("vis_sm",          "Visibilité (miles)"),
            ("wind_vel_kts",    "Vent (nœuds)"),
            ("wind_dir_deg",    "Direction vent (°)"),
            ("sky_ceil_ht",     "Plafond nuageux (ft)"),
            ("temp",            "Température (°C)"),
            ("acft_make",       "Constructeur aéronef"),
            ("acft_model",      "Modèle aéronef"),
            ("acft_category",   "Catégorie aéronef"),
            ("num_eng",         "Nombre de moteurs"),
            ("phase_flt_spec",  "Phase du vol"),
            ("type_fly",        "Type de vol"),
            ("far_part",        "Réglementation FAR"),
            ("damage",          "Dommages aéronef"),
            ("inj_tot_f",       "Blessés fatals (total)"),
            ("inj_tot_s",       "Blessés graves (total)"),
        ]

        lines = []
        for key, label in priority_keys:
            value = features.get(key)
            if value is not None and str(value).strip() not in ("", "nan", "NaN"):
                lines.append(f"• {label:<30} : {value}")

        # Si trop peu de features connues, on ajoute un avertissement
        if len(lines) < 5:
            lines.append(
                "• [Avertissement] Données contextuelles limitées — "
                "raisonnement fondé principalement sur la prédiction ML."
            )

        return "\n".join(lines) if lines else "• Aucune feature contextuelle disponible."

    # ── Parsing de la réponse JSON Gemini ────────────────────────────────────

    @staticmethod
    def _parse_gemini_response(raw_text: str) -> dict[str, Any]:
        """
        Parse la réponse JSON de Gemini de façon robuste.
        Gère les cas où Gemini ajoute des balises ```json malgré l'instruction.

        Args:
            raw_text: Texte brut retourné par Gemini.

        Returns:
            Dict Python parsé.

        Raises:
            ValueError: Si le JSON est invalide après nettoyage.
        """
        import json

        cleaned = raw_text.strip()

        # Nettoyage des balises Markdown si Gemini les ajoute quand même
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Supprime la première ligne (```json ou ```) et la dernière (```)
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "Échec du parsing JSON Gemini. Réponse brute : %s", raw_text[:500]
            )
            raise ValueError(
                f"Gemini n'a pas retourné un JSON valide : {exc}"
            ) from exc

    # ── Point d'entrée principal ──────────────────────────────────────────────

    def generate_safety_report(
        self,
        prediction: str,
        uncertainty_set: list[str],
        accident_features: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Génère un rapport de sécurité structuré via Gemini.

        Args:
            prediction        : Classe majoritaire MAPIE (ex: "MINR")
            uncertainty_set   : Classes plausibles MAPIE (ex: ["MINR","SERS"])
            accident_features : Features brutes du vol analysé

        Returns:
            Dict structuré avec les clés :
              - risk_summary            (str)
              - contributing_factors    (list[str])
              - safety_recommendations  (list[str])
              - worst_case_preparedness (str)
              - confidence_note         (str)
              - _meta                   (dict) : modèle, worst_case utilisé

        Raises:
            ValueError  : Payload invalide ou JSON Gemini non parseable.
            RuntimeError: Erreur API Gemini.
        """
        if not prediction:
            raise ValueError("La prédiction ne peut pas être vide.")

        worst_case = self._worst_case_scenario(uncertainty_set)
        prompt = self._build_prompt(prediction, uncertainty_set, accident_features)

        logger.info(
            "Génération rapport Gemini — prédiction: %s | worst_case: %s",
            prediction, worst_case,
        )

        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,       # Faible température = réponses factuelles
                    max_output_tokens=1024,
                    candidate_count=1,
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Erreur lors de l'appel API Gemini : {exc}"
            ) from exc

        raw_text: str = response.text
        report = self._parse_gemini_response(raw_text)

        # Enrichissement avec métadonnées internes
        report["_meta"] = {
            "model_used": self._model_name,
            "majority_prediction": prediction,
            "worst_case_used": worst_case,
            "uncertainty_set": uncertainty_set,
        }

        return report


# ── Instance singleton ────────────────────────────────────────────────────────
_api_key    = os.getenv("GEMINI_API_KEY", "")
_model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

try:
    gemini_service = GeminiService(api_key=_api_key, model_name=_model_name)
except (ValueError, RuntimeError) as e:
    logger.error("IMPOSSIBLE D'INITIALISER GEMINI : %s", e)
    gemini_service = None  # type: ignore[assignment]