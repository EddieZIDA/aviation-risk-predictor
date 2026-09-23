"""
Test de parité notebook ↔ backend.

Garantit que le backend transforme une ligne du dataset EXACTEMENT comme le
NB03 l'a fait pendant l'entraînement : si ce test échoue, les prédictions
servies par l'API ne correspondent plus au modèle évalué dans les notebooks.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import ARTIFACTS_DIR, requires_artifacts
from utils.feature_engineering import (
    FEATURE_COLS,
    INJURY_ORDER,
    TARGET_COL,
    TEST_START_YEAR,
    TOP_MAKES,
    VAL_START_YEAR,
    compute_top_makes,
    engineer_features,
    select_model_features,
    split_labels,
    split_positions,
)

pytestmark = requires_artifacts

OUTPUTS = ARTIFACTS_DIR / "outputs"
N_ROWS_UNIT = 200  # lignes testées une par une via le chemin « API »


@pytest.fixture(scope="module")
def x_test_expected():
    return pd.read_csv(OUTPUTS / "X_test.csv")


@pytest.fixture(scope="module")
def test_rows(clean_df):
    """Lignes brutes du CSV nettoyé qui composent le jeu de test du NB03."""
    _, _, idx_test = split_positions(clean_df["ev_year"])
    return clean_df.iloc[idx_test]


def test_split_is_temporal(clean_df):
    split = split_labels(clean_df["ev_year"])
    years = clean_df["ev_year"]
    assert years[split == "train"].max() < VAL_START_YEAR
    assert years[split == "val"].between(VAL_START_YEAR, TEST_START_YEAR - 1).all()
    assert years[split == "test"].min() >= TEST_START_YEAR
    shares = split.value_counts(normalize=True)
    assert shares["train"] > 0.5 and shares["test"] > 0.15


def test_top_makes_match_dataset(clean_df):
    assert set(compute_top_makes(clean_df)) == set(TOP_MAKES)


def test_feature_cols_match_preprocessor(service):
    assert list(service._preprocessor.feature_names_in_) == FEATURE_COLS


def test_split_reproduces_notebook(clean_df, test_rows):
    y_test = pd.read_csv(OUTPUTS / "y_test.csv").squeeze()
    expected = test_rows[TARGET_COL].map(INJURY_ORDER).to_numpy()
    np.testing.assert_array_equal(expected, y_test.to_numpy())


def test_batch_pipeline_reproduces_x_test(service, test_rows, x_test_expected):
    X = select_model_features(engineer_features(test_rows))
    got = service._vt.transform(service._preprocessor.transform(X))
    np.testing.assert_allclose(got, x_test_expected.to_numpy(), rtol=1e-9, atol=1e-9)


def _as_mongo_doc(row: pd.Series) -> dict:
    """Simule un document MongoDB / un JSON : NaN → None, types Python natifs."""
    return {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v)
            for k, v in row.items()}


def test_api_path_reproduces_x_test(service, test_rows, x_test_expected):
    for i in range(N_ROWS_UNIT):
        got = service.transform(_as_mongo_doc(test_rows.iloc[i]))
        np.testing.assert_allclose(
            got.to_numpy()[0], x_test_expected.iloc[i].to_numpy(),
            rtol=1e-9, atol=1e-9, err_msg=f"ligne de test n°{i}",
        )


def test_api_path_accepts_json_strings(service, test_rows, x_test_expected):
    """Un CSV importé côté frontend envoie tout en texte : le résultat doit être identique."""
    doc = {k: (None if v is None else str(v)) for k, v in _as_mongo_doc(test_rows.iloc[0]).items()}
    got = service.transform(doc)
    np.testing.assert_allclose(got.to_numpy()[0], x_test_expected.iloc[0].to_numpy(), rtol=1e-9, atol=1e-9)


def test_api_predictions_match_notebook_model(service, test_rows, x_test_expected):
    n = 50
    expected = service._model.predict(x_test_expected.iloc[:n], alpha=0.1)[0] if service.is_mapie \
        else service._model.predict(x_test_expected.iloc[:n])
    labels = {0: "NONE", 1: "MINR", 2: "SERS", 3: "FATL"}
    got = [service.predict_accident(_as_mongo_doc(test_rows.iloc[i]))["prediction"] for i in range(n)]
    assert got == [labels[int(v)] for v in expected]
