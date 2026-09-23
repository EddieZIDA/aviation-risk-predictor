"""
Tests unitaires du typage des payloads et du service de prédiction.
"""

import numpy as np
import pytest

from conftest import requires_artifacts
from utils.feature_engineering import (
    FEATURE_COLS,
    RAW_INPUT_COLS,
    build_model_input,
    payload_to_frame,
)

# ── payload_to_frame / build_model_input ─────────────────────────────────────

def test_empty_payload_gives_all_features():
    X = build_model_input({})
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == 1


def test_missing_acft_make_does_not_crash():
    """Régression : .str sur une colonne entièrement vide levait AttributeError."""
    X = build_model_input({"ev_year": 2015})
    assert X.loc[0, "acft_make_grouped"] == "OTHER"


def test_make_is_normalized_and_aliased():
    X = build_model_input({"acft_make": "  cirrus design corp "})
    assert X.loc[0, "acft_make_grouped"] == "CIRRUS"


def test_categories_are_kept_as_strings():
    """'091' ne doit pas devenir 91 (catégorie inconnue pour l'encodeur)."""
    frame = payload_to_frame({"far_part": "091"})
    assert frame.loc[0, "far_part"] == "091"


def test_none_is_a_category_not_a_missing_value():
    frame = payload_to_frame({"sky_cond_ceil": "NONE"})
    assert frame.loc[0, "sky_cond_ceil"] == "NONE"


def test_numeric_strings_are_parsed():
    frame = payload_to_frame({"wx_temp_c": "21.5", "crew_age": ""})
    assert frame.loc[0, "wx_temp_c"] == 21.5
    assert np.isnan(frame.loc[0, "crew_age"])


def test_invalid_numeric_raises_value_error():
    with pytest.raises(ValueError, match="wx_temp_c"):
        payload_to_frame({"wx_temp_c": "chaud"})


@pytest.mark.parametrize("raw, expected", [
    (True, 1.0), ("True", 1.0), ("false", 0.0), (0, 0.0), ("Y", 1.0),
])
def test_boolean_parsing(raw, expected):
    assert payload_to_frame({"pilot_flying": raw}).loc[0, "pilot_flying"] == expected


def test_afm_missing_flag_is_derived_when_absent():
    assert payload_to_frame({"afm_hrs": 1200}).loc[0, "is_missing_afm_hrs"] == 0.0
    assert payload_to_frame({}).loc[0, "is_missing_afm_hrs"] == 1.0


def test_missing_indicators_are_not_model_features():
    """Régression : is_missing_* poussait vers FATL toute saisie incomplète."""
    assert not any(c.startswith("is_missing_") for c in FEATURE_COLS)


def test_explicit_missing_flag_is_kept():
    frame = payload_to_frame({"afm_hrs": 1200, "is_missing_afm_hrs": 1})
    assert frame.loc[0, "is_missing_afm_hrs"] == 1.0


def test_season_derived_from_month():
    assert payload_to_frame({"ev_month": 7}).loc[0, "ev_season"] == "summer"


def test_unknown_keys_are_ignored():
    frame = payload_to_frame({"inj_tot_f": 3, "damage": "DEST"})
    assert "inj_tot_f" not in frame.columns
    assert list(frame.columns) == RAW_INPUT_COLS


# ── PredictionService (nécessite les artefacts) ──────────────────────────────

@requires_artifacts
def test_prediction_shape(service):
    result = service.predict_accident({"acft_make": "CESSNA", "ev_month": 7})
    assert result["prediction"] in {"NONE", "MINR", "SERS", "FATL"}
    assert set(result["uncertainty_set"]) <= {"NONE", "MINR", "SERS", "FATL"}
    assert result["confidence_level"] == pytest.approx(0.9)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


@requires_artifacts
def test_prediction_rejects_empty_payload(service):
    with pytest.raises(ValueError):
        service.predict_accident({})
