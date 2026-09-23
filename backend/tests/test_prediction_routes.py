"""
Tests HTTP des routes Flask (client de test, sans serveur).
Gemini est remplacé par un faux service : aucun appel réseau n'est fait.
"""

import pytest

from app import create_app
from conftest import requires_artifacts


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ── Santé et erreurs génériques ──────────────────────────────────────────────

def test_health_reports_components(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "online"
    assert {"model_loaded", "mongodb", "gemini_enabled"} <= set(body)


def test_unknown_route_returns_json_404(client):
    res = client.get("/api/nope")
    assert res.status_code == 404
    assert res.get_json()["status"] == "error"


# ── /api/predict ─────────────────────────────────────────────────────────────

def test_predict_rejects_non_json(client):
    res = client.post("/api/predict", data="x", content_type="text/plain")
    assert res.status_code in (400, 503)


@requires_artifacts
def test_predict_rejects_empty_object(client):
    res = client.post("/api/predict", json={})
    assert res.status_code == 400


@requires_artifacts
def test_predict_rejects_bad_number(client):
    res = client.post("/api/predict", json={"wx_temp_c": "chaud"})
    assert res.status_code == 400
    assert res.get_json()["code"] == "VALIDATION_ERROR"


@requires_artifacts
def test_predict_success(client):
    res = client.post("/api/predict", json={"acft_make": "CESSNA", "ev_year": 2018, "ev_month": 7})
    assert res.status_code == 200
    body = res.get_json()
    assert body["prediction"] in {"NONE", "MINR", "SERS", "FATL"}
    assert body["input_coverage"]["provided"] == 3
    assert body["model"]["type"]


def test_predict_schema(client):
    body = client.get("/api/predict/schema").get_json()
    assert "far_part" in body["fields"]["categorical"]


# ── /api/historical ──────────────────────────────────────────────────────────

def test_distribution_requires_whitelisted_field(client):
    res = client.get("/api/historical/distribution?field=$where")
    assert res.status_code in (400, 503)


# ── /api/report ──────────────────────────────────────────────────────────────

class _FakeGemini:
    def generate_safety_report(self, prediction, uncertainty_set, accident_features):
        return {"risk_summary": "ok", "_meta": {"worst_case_used": uncertainty_set[-1]}}


@pytest.fixture
def fake_gemini(monkeypatch):
    import routes.report
    monkeypatch.setattr(routes.report, "gemini_service", _FakeGemini())


def test_report_validates_body(client, fake_gemini):
    res = client.post("/api/report", json={"prediction": "MINR"})
    assert res.status_code == 400


def test_report_success(client, fake_gemini):
    res = client.post("/api/report", json={
        "prediction": "MINR",
        "uncertainty_interval": ["MINR", "SERS"],
        "accident_features": {"acft_make": "CESSNA"},
    })
    assert res.status_code == 200
    assert res.get_json()["report"]["_meta"]["worst_case_used"] == "SERS"
