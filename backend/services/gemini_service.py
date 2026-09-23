"""
services/gemini_service.py — Service de Génération de Rapports LLM
====================================================================
Responsabilité UNIQUE : construire le prompt de sécurité aéronautique
et appeler l'API Google Gemini. Aucune logique de routing ici.

Principe directeur du prompt :
  Gemini doit toujours raisonner à partir du SCÉNARIO LE PLUS GRAVE
  présent dans l'uncertainty_set MAPIE, pas seulement de la prédiction
  majoritaire. C'est la garantie d'une recommandation conservatrice.

SDK : google-genai (remplace google-generativeai, déprécié). La réponse est
contrainte par un schéma Pydantic (sortie structurée) puis revalidée ici.
"""

import json
import logging
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

from config.settings import settings

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


class SafetyReport(BaseModel):
    """Schéma imposé à Gemini (response_schema) : le JSON renvoyé doit le respecter."""

    risk_summary: str = Field(description="Synthèse exécutive en 2-3 phrases.")
    contributing_factors: list[str] = Field(description="3 à 5 facteurs de risque, sans puce ni numéro.")
    safety_recommendations: list[str] = Field(description="3 à 5 recommandations, sans puce ni numéro.")
    worst_case_preparedness: str = Field(description="Préparation au scénario le plus grave.")
    confidence_note: str = Field(description="Explication de l'intervalle MAPIE pour un pilote.")


# Clés que Gemini doit obligatoirement renvoyer (revalidées après parsing)
REPORT_KEYS: dict[str, type] = {
    "risk_summary": str,
    "contributing_factors": list,
    "safety_recommendations": list,
    "worst_case_preparedness": str,
    "confidence_note": str,
}

# Variables décrivant le contexte du vol, transmises au LLM.
# Les variables connues seulement APRÈS l'accident (dommages, nombre de blessés...)
# sont volontairement exclues : elles donneraient la réponse au modèle.
PROMPT_FEATURES: list[tuple[str, str]] = [
    ("ev_state",        "État (USA)"),
    ("ev_year",         "Année"),
    ("ev_month",        "Mois"),
    ("light_cond",      "Conditions lumineuses"),
    ("wx_cond_basic",   "Conditions météo (VMC/IMC)"),
    ("vis_km",          "Visibilité (km)"),
    ("wind_vel_kts",    "Vent (nœuds)"),
    ("gust_kts",        "Rafales (nœuds)"),
    ("sky_ceil_ht_m",   "Plafond nuageux (m)"),
    ("wx_temp_c",       "Température (°C)"),
    ("acft_make",       "Constructeur"),
    ("acft_category",   "Catégorie d'aéronef"),
    ("num_eng",         "Nombre de moteurs"),
    ("acft_year",       "Année de construction"),
    ("homebuilt",       "Construction amateur"),
    ("type_fly",        "Type de vol"),
    ("far_part",        "Réglementation FAR"),
    ("flt_plan_filed",  "Plan de vol déposé"),
    ("crew_category",   "Rôle du pilote"),
    ("crew_age",        "Âge du pilote"),
    ("afm_hrs",         "Heures de vol de l'aéronef (cellule, pas le pilote)"),
]


class GeminiService:
    """
    Service singleton pour la génération de rapports de sécurité
    aéronautique via l'API Google Gemini.
    """

    def __init__(self, api_key: str, model_name: str, timeout_s: float = 30,
                 fallback_model: str | None = None) -> None:
        """
        Configure le client Gemini.

        Args:
            api_key   : Clé API Google AI Studio.
            model_name: Identifiant du modèle (variable GEMINI_MODEL).
            timeout_s : Délai maximal d'un appel, en secondes.
            fallback_model: Modèle utilisé si le principal reste indisponible (503/429).

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
            self._client = genai.Client(api_key=api_key)
            self._model_name = model_name
            self._fallback_model = fallback_model if fallback_model != model_name else None
            self._config = types.GenerateContentConfig(
                temperature=0.3,                        # réponses factuelles
                # Les modèles « à réflexion » (ex. gemini-2.5-flash) décomptent leurs jetons de
                # réflexion de cette limite : à 2048, le JSON était parfois coupé en plein milieu.
                max_output_tokens=8192,
                response_mime_type="application/json",  # JSON natif, sans Markdown
                response_schema=SafetyReport,           # structure garantie par l'API
                http_options=types.HttpOptions(
                    timeout=int(timeout_s * 1000),  # en ms
                    # 429 (quota) et 503 (« high demand ») sont fréquents et transitoires :
                    # 3 tentatives avec attente exponentielle (1 s, 2 s...) avant d'abandonner.
                    retry_options=types.HttpRetryOptions(
                        attempts=3, initial_delay=1.0, max_delay=8.0,
                        http_status_codes=[429, 500, 502, 503, 504],
                    ),
                ),
                # Pas d'outils : désactive l'appel de fonctions automatique (et son avertissement)
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
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

    @staticmethod
    def _reasoning_instructions(
        prediction: str, uncertainty_set: list[str], worst_case: str,
    ) -> tuple[str, str]:
        """
        Adapte la consigne au cas rencontré, pour éviter un discours de « précaution »
        hors de propos quand le seul scénario plausible est bénin.

        Returns:
            (consigne de raisonnement, contenu attendu pour worst_case_preparedness)
        """
        if not uncertainty_set:
            return (
                "Le modèle ne peut désigner aucune classe avec 90 % de confiance : la situation "
                "est atypique. Par précaution, raisonne sur le scénario le plus grave (FATL).",
                "comment se préparer à un accident grave, puisque rien ne permet de l'exclure.",
            )
        if worst_case == "NONE":
            return (
                "Le modèle est confiant : seul le scénario NONE (aucun blessé) est plausible. "
                "Le rapport doit le dire clairement et se concentrer sur les points de vigilance "
                "qui maintiennent ce faible niveau de risque, sans dramatiser.",
                "les points de vigilance qui permettent de garder ce vol dans le scénario NONE "
                "(ce qui pourrait faire basculer la situation et comment l'éviter).",
            )
        if worst_case == prediction and len(uncertainty_set) == 1:
            return (
                f"Le modèle est confiant : seul le scénario {prediction} est plausible à 90 %. "
                "Rédige le rapport pour ce scénario.",
                f"comment se préparer concrètement au scénario {prediction}.",
            )
        if worst_case == prediction:
            return (
                f"La classe la plus probable ({prediction}) est aussi la plus grave de l'ensemble "
                "plausible : des issues moins graves restent possibles, mais le rapport doit être "
                f"rédigé pour {prediction}.",
                f"comment se préparer concrètement au scénario {prediction}.",
            )
        return (
            f"La classe la plus probable est {prediction}, MAIS le scénario plus grave {worst_case} "
            "reste plausible à 90 % de confiance. Principe de précaution aéronautique : les "
            f"recommandations doivent préparer au scénario {worst_case}, sans minimiser le risque.",
            f"comment se préparer spécifiquement au scénario {worst_case}, même si la classe la "
            f"plus probable est {prediction}.",
        )

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
        """
        worst_case = self._worst_case_scenario(uncertainty_set)
        key_features = self._format_key_features(accident_features)

        uncertainty_str = ", ".join(
            f"{label} ({RISK_DESCRIPTIONS.get(label, label)})" for label in uncertainty_set
        ) if uncertainty_set else "aucune classe (ensemble vide)"

        reasoning, preparedness_hint = self._reasoning_instructions(prediction, uncertainty_set, worst_case)

        prompt = f"""Tu es AERO-ANALYST, un expert en sécurité aéronautique de niveau FAA/OACI.
Tu rédiges, à partir d'une évaluation statistique, un rapport de sécurité factuel, sobre et
exploitable par un pilote ou un instructeur. Tu écris en français, sans jargon statistique inutile.

══════════════════════════════════════════
CONTEXTE DU VOL
══════════════════════════════════════════
{key_features}

══════════════════════════════════════════
ÉVALUATION DU RISQUE (modèle XGBoost + prédiction conforme MAPIE)
══════════════════════════════════════════
• Classe la plus probable : {prediction} — {RISK_DESCRIPTIONS.get(prediction, prediction)}
• Classes plausibles à 90 % : {uncertainty_str}
• Scénario de référence du rapport : {worst_case} — {RISK_DESCRIPTIONS.get(worst_case, worst_case)}

══════════════════════════════════════════
RAISONNEMENT ATTENDU
══════════════════════════════════════════
{reasoning}

Règles de rédaction :
- Appuie chaque facteur de risque sur une donnée précise du contexte (valeur, condition).
  N'invente aucune donnée absente du contexte.
- Recommandations concrètes et actionnables, adaptées au niveau de risque : pas de ton
  alarmiste si le risque est faible, aucune minimisation s'il est élevé.
- Listes : 3 à 5 éléments, sans numéro, tiret ni puce en début de phrase
  (OUI : "Vérifier la pression hydraulique avant le décollage." NON : "1. Vérifier...").
- Écris les pourcentages sous la forme « 90 % ».

══════════════════════════════════════════
CONTENU DE CHAQUE CHAMP
══════════════════════════════════════════
- risk_summary : 2 à 3 phrases ; niveau de risque estimé et ce qu'il implique pour ce vol.
- contributing_factors : facteurs du contexte qui augmentent (ou limitent) le risque.
- safety_recommendations : actions pour le pilote ou l'exploitant.
- worst_case_preparedness : {preparedness_hint}
- confidence_note : explication pour un pilote non statisticien de ce que signifie ici
  l'ensemble « {', '.join(uncertainty_set) or 'vide'} » (2 à 3 phrases). Sens exact : sur
  l'ensemble des vols évalués de cette façon, la gravité réelle fait partie de l'ensemble annoncé
  environ 9 fois sur 10 ; plus l'ensemble contient de classes, plus le modèle est incertain pour
  ce vol. Ne parle pas de « répéter ce vol ».
- N'interprète jamais les heures de vol de l'aéronef comme l'expérience du pilote.

Réponds UNIQUEMENT avec l'objet JSON demandé."""

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
        lines = []
        for key, label in PROMPT_FEATURES:
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

    # ── Parsing et validation de la réponse Gemini ───────────────────────────

    @staticmethod
    def _parse_gemini_response(raw_text: str) -> dict[str, Any]:
        """
        Parse et valide le JSON renvoyé par Gemini.
        Tolère des balises ```json autour du contenu malgré l'instruction.

        Raises:
            RuntimeError: JSON invalide ou clés manquantes (erreur du service
                          externe → 502, pas une erreur du client).
        """
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1]).strip()

        try:
            report = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("JSON Gemini invalide. Début de la réponse : %s", raw_text[:500])
            raise RuntimeError("Gemini n'a pas renvoyé un JSON valide.") from exc

        missing = [k for k, t in REPORT_KEYS.items() if not isinstance(report.get(k), t)]
        if missing:
            raise RuntimeError(f"Rapport Gemini incomplet (clés manquantes : {missing}).")
        return report

    # ── Appel API avec modèle de secours ─────────────────────────────────────

    def _generate(self, prompt: str) -> tuple[dict[str, Any], str]:
        """
        Essaie le modèle principal (avec les nouvelles tentatives du SDK) puis le modèle
        de secours. Un modèle surchargé (429/5xx) ou une réponse vide/tronquée/invalide
        fait passer au suivant.

        Returns:
            (rapport validé, nom du modèle qui a répondu)
        Raises:
            RuntimeError: aucun modèle n'a produit de rapport exploitable.
        """
        models = [self._model_name] + ([self._fallback_model] if self._fallback_model else [])
        last_error: Exception | None = None
        for model in models:
            try:
                response = self._client.models.generate_content(
                    model=model, contents=prompt, config=self._config,
                )
            except errors.APIError as exc:
                last_error = exc
                if exc.code not in (429, 500, 502, 503, 504):
                    break  # erreur non transitoire (clé invalide, modèle inconnu...) : inutile d'insister
                logger.warning("Modèle '%s' indisponible (%s)", model, exc.code)
                continue
            except Exception as exc:  # réseau, timeout...
                last_error = exc
                logger.warning("Modèle '%s' injoignable : %s", model, exc)
                continue

            finish = response.candidates[0].finish_reason if response.candidates else None
            if not response.text:
                last_error = RuntimeError(f"réponse vide (raison : {finish})")
                logger.warning("Modèle '%s' : réponse vide (%s)", model, finish)
                continue
            try:
                return self._parse_gemini_response(response.text), model
            except RuntimeError as exc:
                last_error = exc
                logger.warning("Modèle '%s' : rapport inexploitable (fin : %s)", model, finish)
        raise RuntimeError(f"Échec de la génération du rapport : {last_error}") from last_error

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
            Dict avec les clés de REPORT_KEYS + `_meta` (modèle, worst_case utilisé).

        Raises:
            ValueError  : Entrée invalide (→ 400).
            RuntimeError: Échec de l'appel Gemini ou réponse inexploitable (→ 502).
        """
        if prediction not in SEVERITY_ORDER:
            raise ValueError(f"Prédiction inconnue : {prediction!r}.")
        unknown = [label for label in uncertainty_set if label not in SEVERITY_ORDER]
        if unknown:
            raise ValueError(f"Classes inconnues dans l'intervalle : {unknown}.")

        worst_case = self._worst_case_scenario(uncertainty_set)
        prompt = self._build_prompt(prediction, uncertainty_set, accident_features)
        logger.info("Rapport Gemini — prédiction: %s | worst_case: %s", prediction, worst_case)

        report, model_used = self._generate(prompt)
        report["_meta"] = {
            "model_used": model_used,
            "majority_prediction": prediction,
            "worst_case_used": worst_case,
            "uncertainty_set": uncertainty_set,
        }
        return report


# ── Instance singleton ────────────────────────────────────────────────────────
try:
    gemini_service: GeminiService | None = GeminiService(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
        timeout_s=settings.gemini_timeout_s,
        fallback_model=settings.gemini_fallback_model or None,
    )
except (ValueError, RuntimeError) as e:
    logger.error("IMPOSSIBLE D'INITIALISER GEMINI : %s", e)
    gemini_service = None
