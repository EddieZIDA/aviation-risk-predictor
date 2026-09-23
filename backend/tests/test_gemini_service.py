"""
Tests de la logique locale du service Gemini (prompt, parsing) — sans appel réseau.
"""

import json

import pytest

from services.gemini_service import GeminiService

VALID_REPORT = {
    "risk_summary": "Résumé.",
    "contributing_factors": ["Vent fort."],
    "safety_recommendations": ["Reporter le vol."],
    "worst_case_preparedness": "Préparation.",
    "confidence_note": "Note.",
}


@pytest.mark.parametrize("labels, expected", [
    (["MINR", "SERS"], "SERS"),
    (["NONE"], "NONE"),
    (["FATL", "NONE"], "FATL"),
    ([], "FATL"),  # ensemble vide → hypothèse la plus prudente
])
def test_worst_case_scenario(labels, expected):
    assert GeminiService._worst_case_scenario(labels) == expected


def test_parse_accepts_markdown_fence():
    raw = "```json\n" + json.dumps(VALID_REPORT) + "\n```"
    assert GeminiService._parse_gemini_response(raw)["risk_summary"] == "Résumé."


def test_parse_rejects_invalid_json():
    with pytest.raises(RuntimeError):
        GeminiService._parse_gemini_response("pas du json")


def test_parse_rejects_incomplete_report():
    with pytest.raises(RuntimeError, match="confidence_note"):
        GeminiService._parse_gemini_response(json.dumps({**VALID_REPORT, "confidence_note": None}))


def test_prompt_never_leaks_outcome_variables():
    """Les variables connues après l'accident ne doivent pas atteindre le LLM."""
    features = {"acft_make": "CESSNA", "damage": "DEST", "inj_tot_f": 2, "inj_tot_s": 1}
    text = GeminiService._format_key_features(features)
    assert "CESSNA" in text
    assert "DEST" not in text and "inj_tot" not in text


def test_prompt_is_not_alarmist_when_only_none_is_plausible():
    reasoning, preparedness = GeminiService._reasoning_instructions("NONE", ["NONE"], "NONE")
    assert "MAIS" not in reasoning and "sans dramatiser" in reasoning
    assert "vigilance" in preparedness


def test_prompt_escalates_when_a_worse_class_is_plausible():
    reasoning, preparedness = GeminiService._reasoning_instructions("MINR", ["MINR", "FATL"], "FATL")
    assert "MAIS" in reasoning and "FATL" in preparedness


def test_falls_back_to_second_model_when_primary_is_overloaded():
    """503 persistant sur le modèle principal → le modèle de secours répond."""
    from types import SimpleNamespace

    from google.genai import errors

    svc = GeminiService(api_key="test", model_name="primary", fallback_model="backup")
    calls = []

    def fake_generate(model, contents, config):
        calls.append(model)
        if model == "primary":
            raise errors.ServerError(503, {"error": {"code": 503, "message": "high demand"}})
        return SimpleNamespace(text=json.dumps(VALID_REPORT), candidates=[])

    svc._client = SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate))
    report = svc.generate_safety_report("MINR", ["MINR"], {"acft_make": "CESSNA"})
    assert calls == ["primary", "backup"]
    assert report["_meta"]["model_used"] == "backup"


def test_truncated_json_falls_back_to_next_model():
    """Régression : un JSON coupé (limite de jetons atteinte) doit passer au modèle suivant."""
    from types import SimpleNamespace

    svc = GeminiService(api_key="test", model_name="primary", fallback_model="backup")
    truncated = json.dumps(VALID_REPORT)[:60]

    def fake_generate(model, contents, config):
        text = truncated if model == "primary" else json.dumps(VALID_REPORT)
        return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")])

    svc._client = SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate))
    report = svc.generate_safety_report("FATL", ["FATL"], {})
    assert report["_meta"]["model_used"] == "backup"
