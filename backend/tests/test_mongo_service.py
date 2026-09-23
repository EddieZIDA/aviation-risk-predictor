"""
Tests du garde-fou anti-fuite et des requêtes MongoDB.

Les tests d'intégration sont ignorés si aucun serveur MongoDB n'est joignable.
"""

import math

import pytest

from services.mongo_service import LEAKAGE_GUARD, MongoService, _json_safe, mongo_service

requires_mongo = pytest.mark.skipif(mongo_service is None, reason="MongoDB non joignable.")


# ── Unitaires ────────────────────────────────────────────────────────────────

def test_safe_filter_always_contains_guard():
    assert MongoService._safe_filter() == LEAKAGE_GUARD
    assert MongoService._safe_filter({"ev_state": "TX"}) == {**LEAKAGE_GUARD, "ev_state": "TX"}


def test_safe_filter_cannot_be_overridden():
    with pytest.raises(ValueError):
        MongoService._safe_filter({"split": "test"})


def test_json_safe_replaces_non_finite_floats():
    doc = _json_safe({"a": math.nan, "b": math.inf, "c": 1.5, "d": "x"})
    assert doc == {"a": None, "b": None, "c": 1.5, "d": "x"}


# ── Intégration (base locale chargée par seed_mongo.py) ──────────────────────

@requires_mongo
def test_collection_has_split_field():
    assert mongo_service.has_split_field(), "Relancez seed_mongo.py (champ 'split' absent)."


@requires_mongo
def test_sample_never_contains_test_rows():
    docs = mongo_service.get_accidents_sample(limit=500)
    assert docs and all("split" not in d for d in docs)


@requires_mongo
def test_stats_exclude_test_split():
    total_all = mongo_service._accidents.count_documents({})
    total_test = mongo_service._accidents.count_documents({"split": "test"})
    assert mongo_service.get_stats_summary()["total_accidents"] == total_all - total_test


@requires_mongo
def test_random_example_only_returns_requested_fields():
    doc = mongo_service.get_random_example(["ev_state", "acft_make"])
    assert doc is not None and set(doc) <= {"ev_state", "acft_make"}


@requires_mongo
def test_severity_by_field_sums_to_total():
    for row in mongo_service.get_severity_by("light_cond"):
        assert row["NONE"] + row["MINR"] + row["SERS"] + row["FATL"] == row["total"]
