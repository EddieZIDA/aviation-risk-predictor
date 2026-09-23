"""
utils/feature_engineering.py — Source unique du feature engineering
====================================================================
Ce module est partagé par :
  - notebooks/03_feature_engineering.ipynb  (entraînement)
  - backend/services/prediction_service.py  (inférence)
  - seed_mongo.py                           (attribution du champ `split`)

Garder une seule implémentation évite le « training/serving skew » : le modèle
voit exactement les mêmes transformations à l'entraînement et en production.
Le test tests/test_feature_parity.py vérifie que ce module reproduit
bit à bit notebooks/outputs/X_test.csv.

Contrat d'entrée : une ligne au format de data/processed/ntsb_clean_final.csv
(sortie du NB01, identique aux documents de la collection MongoDB `accidents`).
Les unités sont donc DÉJÀ converties en SI par le NB01 (wx_temp_c, vis_km,
rwy_len_m, ...) — aucune conversion d'unité n'est faite ici.

Flux :
  payload JSON ─► payload_to_frame() ─► engineer_features() ─► select_model_features()
                  (typage/nettoyage)     (NB03 §1 à §5)         (NB03 §7)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42  # graine des modèles (le découpage, lui, est temporel et déterministe)

# ── Découpage temporel (NB03 §7) ──────────────────────────────────────────────
# Le modèle est entraîné sur le passé et évalué sur les années les plus récentes,
# comme il le serait en production (prédire des vols futurs).
VAL_START_YEAR = 2019   # train : 2008–2018 (≈ 62 %)
TEST_START_YEAR = 2022  # val   : 2019–2021 (≈ 16 %) · test : 2022 → aujourd'hui (≈ 23 %)

# ── Cible ─────────────────────────────────────────────────────────────────────
TARGET_COL = "ev_highest_injury"
INJURY_ORDER: dict[str, int] = {"NONE": 0, "MINR": 1, "SERS": 2, "FATL": 3}

# ── Regroupement des constructeurs (NB03 §3) ──────────────────────────────────
MAKE_ALIASES: dict[str, str] = {
    "ROBINSON HELICOPTER COMPANY": "ROBINSON", "ROBINSON HELICOPTER": "ROBINSON",
    "CIRRUS DESIGN CORP": "CIRRUS",            "CIRRUS DESIGN": "CIRRUS",
    "DIAMOND AIRCRAFT IND INC": "DIAMOND",     "DIAMOND AIRCRAFT": "DIAMOND",
    "AIR TRACTOR INC": "AIR TRACTOR",          "BOMBARDIER INC": "BOMBARDIER",
    "DEHAVILLAND": "DE HAVILLAND",             "DE HAVILLAND CANADA": "DE HAVILLAND",
}
TOP_N_MAKES = 25

# Les 25 constructeurs les plus fréquents du dataset complet, figés pour
# l'inférence (une seule ligne ne permet pas de recalculer un value_counts).
# Le test de parité vérifie que compute_top_makes(dataset) redonne cette liste.
TOP_MAKES: list[str] = [
    "CESSNA", "PIPER", "BEECH", "BOEING", "ROBINSON", "BELL", "CIRRUS",
    "AIR TRACTOR", "MOONEY", "AIRBUS", "DE HAVILLAND", "SCHWEIZER", "BELLANCA",
    "MAULE", "EMBRAER", "AERONCA", "HUGHES", "DIAMOND", "BOMBARDIER", "VANS",
    "EUROCOPTER", "CHAMPION", "STINSON", "LUSCOMBE", "NORTH AMERICAN",
]

# ── Saisons (identique au NB01) ───────────────────────────────────────────────
SEASON_BY_MONTH: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

DOW_MAP: dict[str, int] = {"Mo": 0, "Tu": 1, "We": 2, "Th": 3, "Fr": 4, "Sa": 5, "Su": 6}

# Colonnes passées en log1p (NB03 §5) ; la colonne source est ensuite abandonnée.
LOG_COLS: list[str] = ["afm_hrs", "cert_max_gr_wt_kg", "fuel_on_board_l", "rwy_len_m"]

# ── Variables exclues : fuite de la cible (NB03 §6) ──────────────────────────
# Renseignées par l'enquête APRÈS l'accident, elles « contiennent » la réponse :
# un modèle qui les utilise ne peut pas servir à évaluer un vol à l'avance.
LEAKAGE_COLS: dict[str, str] = {
    "crew_tox_perf": "test toxicologique de l'équipage, pratiqué lors de l'autopsie "
                     "(96,6 % d'accidents mortels quand il vaut Y)",
    "elt_oper": "la balise de détresse s'est déclenchée : conséquence de l'impact",
    "latlong_acq": "position estimée (EST) à partir de l'épave plutôt que mesurée",
    "wx_src_iic": "source de la météo retenue par l'enquêteur",
}

# ── Groupes de colonnes consommés par le ColumnTransformer (NB03 §6) ──────────
# Ces listes ne contiennent que les colonnes réellement présentes dans le
# dataset : leur concaténation est égale à preprocessor.feature_names_in_.
BINARY_YN_COLS: list[str] = [
    "homebuilt", "second_pilot", "afm_hrs_since", "elt_install",
    "ev_nr_apt_loc", "wind_dir_ind", "wind_vel_ind", "gust_ind", "pilot_flying",
]

NUMERIC_BINARY_COLS: list[str] = [
    "noaa_fog", "noaa_rain", "noaa_snow", "noaa_thunder",
    "adverse_weather", "poor_visibility", "high_wind",
    "extreme_temp", "fog_risk", "is_imc",
]

OHE_COLS: list[str] = [
    "acft_category", "ev_season", "light_cond",
    "sky_cond_nonceil", "sky_cond_ceil",
    "type_last_insp", "crew_category", "med_certf",
    "med_crtf_vldty", "fixed_retractable", "acft_make_grouped", "seat_occ_pic",
]

OHE_HIGH_COLS: list[str] = [
    "ev_state", "far_part", "type_fly", "flt_plan_filed",
    "flight_plan_activated", "pc_profession", "infl_rest_inst",
    "elt_type", "available_restraint", "restraint_used", "med_crtf_limit",
]

ORDINAL_COLS: list[str] = ["crew_sex"]

NUMERIC_COLS: list[str] = [
    "noaa_wind_knots", "noaa_maxwind_knots", "noaa_gust_knots",
    "noaa_slp_hpa", "noaa_visib_km", "noaa_prcp_mm", "noaa_dist_km",
    "gust_excess", "wx_temp_c", "wx_dew_pt_c",
    "wind_dir_deg", "wind_vel_kts", "gust_kts", "vis_km",
    "sky_ceil_ht_m", "sky_nonceil_ht_m", "altimeter_hpa", "td_spread",
    "apt_elev_m", "apt_dist_km", "wx_obs_elev_m", "wx_obs_dist_km",
    "rwy_width_m", "rwy_len_m_log", "num_eng", "fc_seats", "pax_seats",
    "total_seats", "afm_hrs_log", "afm_hrs_last_insp",
    "cert_max_gr_wt_kg_log", "fuel_on_board_l_log",
    "acft_age", "maintenance_ratio", "crew_age",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "ev_year", "wx_obs_time",
]

# Colonnes catégorielles textuelles : leurs espaces parasites sont retirés (§1).
CATEGORICAL_FEATURE_COLS: list[str] = [
    c for c in BINARY_YN_COLS + OHE_COLS + OHE_HIGH_COLS + ORDINAL_COLS
    if c not in {"pilot_flying", "acft_make_grouped", "ev_season"}
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "binary_yn": BINARY_YN_COLS,
    "binary_num": NUMERIC_BINARY_COLS,
    "ohe": OHE_COLS,
    "ohe_high": OHE_HIGH_COLS,
    "ordinal": ORDINAL_COLS,
    "numeric": NUMERIC_COLS,
}

# Colonnes attendues par preprocessing_pipeline.pkl, dans l'ordre.
FEATURE_COLS: list[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]

# ── Colonnes calculées ici (absentes du CSV nettoyé) ──────────────────────────
DERIVED_COLS: set[str] = {
    "adverse_weather", "poor_visibility", "high_wind", "extreme_temp",
    "td_spread", "fog_risk", "is_imc", "maintenance_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "acft_age", "gust_excess", "acft_make_grouped",
    *(f"{c}_log" for c in LOG_COLS),
}

# ── Indicateurs de valeur manquante : exclus du modèle ────────────────────────
# Les colonnes is_missing_* du NB01 sont corrélées à l'issue (le formulaire NTSB
# est moins complet quand le pilote est décédé : 33 % d'accidents mortels quand
# l'âge du pilote manque, 17 % sinon). Pour un vol réel ces informations sont
# connues, l'indicateur vaudrait toujours 0 : le modèle apprendrait « donnée
# absente ⇒ accident grave » et pousserait vers FATL toute saisie incomplète.
# Seul is_missing_afm_hrs est lu, pour la correction d'anomalie du §2.2.
MISSING_INDICATOR_COLS: list[str] = [
    "is_missing_rwy_len", "is_missing_rwy_width", "is_missing_pax_seats",
    "is_missing_crew_age", "is_missing_fuel_on_board", "is_missing_afm_hrs",
    "is_missing_afm_hrs_last_insp", "is_missing_acft_year", "is_missing_wind_vel_kts",
]

# Colonnes sources utilisées uniquement pour calculer des features dérivées.
SOURCE_ONLY_COLS: list[str] = [
    "acft_make", "ev_time", "ev_dow", "ev_month", "wx_cond_basic",
    "afm_hrs", "cert_max_gr_wt_kg", "fuel_on_board_l", "rwy_len_m", "acft_year",
    "is_missing_afm_hrs",
]

# Contrat d'entrée de l'API : colonnes du CSV nettoyé utiles au modèle.
RAW_INPUT_COLS: list[str] = [
    c for c in FEATURE_COLS if c not in DERIVED_COLS
] + SOURCE_ONLY_COLS

# Typage des colonnes brutes (le JSON et les CSV importés arrivent souvent en texte).
BOOL_INPUT_COLS: list[str] = ["noaa_fog", "noaa_rain", "noaa_snow", "noaa_thunder", "pilot_flying"]
NUMERIC_INPUT_COLS: list[str] = [
    c for c in RAW_INPUT_COLS
    if c in NUMERIC_COLS
    or c in {"afm_hrs", "cert_max_gr_wt_kg", "fuel_on_board_l", "rwy_len_m",
             "acft_year", "ev_time", "ev_month", "is_missing_afm_hrs"}
]
CATEGORICAL_INPUT_COLS: list[str] = [
    c for c in RAW_INPUT_COLS if c not in NUMERIC_INPUT_COLS and c not in BOOL_INPUT_COLS
]

_TRUE_STRINGS = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE_STRINGS = {"false", "f", "no", "n", "0", "0.0"}


# ══════════════════════════════════════════════════════════════════════════════
# Transformations NB03
# ══════════════════════════════════════════════════════════════════════════════

def normalize_make(makes: pd.Series) -> pd.Series:
    """Met en majuscules, retire les espaces et applique les alias de constructeurs."""
    upper = makes.map(lambda m: m.upper().strip() if isinstance(m, str) else np.nan)
    return upper.replace(MAKE_ALIASES)


def compute_top_makes(df: pd.DataFrame, n: int = TOP_N_MAKES) -> list[str]:
    """Constructeurs les plus fréquents du dataset (utilisé par le NB03)."""
    return normalize_make(df["acft_make"]).value_counts().head(n).index.tolist()


def engineer_features(df: pd.DataFrame, top_makes: Iterable[str] = TOP_MAKES) -> pd.DataFrame:
    """
    Applique les étapes 1 à 5 du NB03 et retourne une copie enrichie du DataFrame.

    Fonctionne ligne à ligne : le résultat d'une ligne ne dépend jamais des
    autres lignes (condition nécessaire pour l'inférence unitaire).
    """
    df = df.copy()
    top_makes = set(top_makes)

    # §1 Catégories : « UNK », « UNK  » et « UNK\r\n » désignent la même valeur
    for col in CATEGORICAL_FEATURE_COLS:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

    # §2.2 afm_hrs == 0 sans indicateur « manquant » → valeur aberrante → NaN
    mask_zero_hrs = (df["afm_hrs"] == 0) & (~df["is_missing_afm_hrs"].astype(bool))
    df.loc[mask_zero_hrs, "afm_hrs"] = np.nan

    # §2.3 Rafale renseignée mais vent nul → le vent moyen prend la valeur de la rafale
    mask_gust_no_wind = df["noaa_gust_knots"].notna() & (df["noaa_wind_knots"] == 0)
    df.loc[mask_gust_no_wind, "noaa_wind_knots"] = df.loc[mask_gust_no_wind, "noaa_gust_knots"]

    # §3 Constructeurs : top 25 + OTHER (NaN → OTHER)
    df["acft_make"] = normalize_make(df["acft_make"])
    df["acft_make_grouped"] = df["acft_make"].map(lambda m: m if m in top_makes else "OTHER")

    # §4.1 Indicateurs météo
    df["adverse_weather"] = (
        df["noaa_fog"].astype(bool) | df["noaa_rain"].astype(bool)
        | df["noaa_snow"].astype(bool) | df["noaa_thunder"].astype(bool)
    ).astype(int)
    df["poor_visibility"] = (df["vis_km"] < 5).astype(int)          # minima VMC OACI
    df["high_wind"] = (df["noaa_wind_knots"] > 25).astype(int)
    df["extreme_temp"] = ((df["wx_temp_c"] < -20) | (df["wx_temp_c"] > 35)).astype(int)
    df["td_spread"] = df["wx_temp_c"] - df["wx_dew_pt_c"]
    df["fog_risk"] = (df["td_spread"] < 3).astype(int)              # écart T-Td < 3 °C
    df["is_imc"] = (df["wx_cond_basic"] == "IMC").astype(int)

    # §4.2 Part des heures de vol écoulées depuis la dernière inspection
    eps = 1e-3
    df["maintenance_ratio"] = (
        (df["afm_hrs"] - df["afm_hrs_last_insp"]).clip(lower=0) / (df["afm_hrs"] + eps)
    ).clip(0, 1)

    # §4.3 Encodage cyclique heure / jour / mois
    df["ev_hour"] = (df["ev_time"] // 100).clip(0, 23)
    df["hour_sin"] = np.sin(2 * np.pi * df["ev_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["ev_hour"] / 24)
    df["ev_dow_num"] = df["ev_dow"].map(DOW_MAP)
    df["dow_sin"] = np.sin(2 * np.pi * df["ev_dow_num"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["ev_dow_num"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["ev_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["ev_month"] / 12)

    # §4.4 Âge de l'appareil, borné à [0, 80] ans
    df["acft_age"] = (df["ev_year"] - df["acft_year"]).clip(lower=0, upper=80)

    # §4.5 Excès de rafale sur le vent moyen
    df["gust_excess"] = (df["noaa_gust_knots"] - df["noaa_wind_knots"]).clip(lower=0).fillna(0)

    # §5 Log1p des distributions très asymétriques
    for col in LOG_COLS:
        df[f"{col}_log"] = np.log1p(df[col])

    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les FEATURE_COLS au format attendu par le ColumnTransformer (NB03 §7)."""
    X = df[FEATURE_COLS].copy()
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    for col in X.columns:
        if X[col].dtype.name == "string":
            X[col] = X[col].astype(object)
    return X


def split_labels(ev_year: pd.Series) -> pd.Series:
    """
    Découpage temporel du NB03 : 'train' avant VAL_START_YEAR, 'val' jusqu'à
    TEST_START_YEAR exclu, 'test' ensuite. Retourne une Series alignée sur ev_year.
    """
    labels = np.select(
        [ev_year < VAL_START_YEAR, ev_year < TEST_START_YEAR],
        ["train", "val"],
        default="test",
    )
    return pd.Series(labels, index=ev_year.index, name="split")


def split_positions(ev_year: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions (iloc) des lignes train / val / test, dans l'ordre d'origine du dataset."""
    labels = split_labels(ev_year).to_numpy()
    return tuple(np.flatnonzero(labels == name) for name in ("train", "val", "test"))


# ══════════════════════════════════════════════════════════════════════════════
# Inférence : payload JSON → DataFrame prêt pour le preprocessor
# ══════════════════════════════════════════════════════════════════════════════

def _is_missing(value: Any) -> bool:
    """None, NaN ou chaîne vide. « NONE » n'est PAS manquant : c'est une catégorie NTSB."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return isinstance(value, float) and np.isnan(value)


def _to_bool(value: Any) -> float:
    """Booléen tolérant (True, 'true', 'Y', 1...) → 1.0 / 0.0 / NaN."""
    if _is_missing(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return 1.0
    if text in _FALSE_STRINGS:
        return 0.0
    raise ValueError(f"Valeur booléenne invalide : {value!r}")


def _to_float(col: str, value: Any) -> float:
    if _is_missing(value) or (isinstance(value, str) and value.strip().lower() in {"nan", "null"}):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{col}' doit être numérique (reçu : {value!r}).") from None


def payload_to_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """
    Convertit un payload JSON (format CSV nettoyé / document MongoDB) en
    DataFrame d'une ligne correctement typée.

    - Les clés hors RAW_INPUT_COLS sont ignorées.
    - "", None (et "nan"/"null" pour les numériques) → valeur manquante,
      imputée ensuite par le preprocessor.
    - Les catégories restent des chaînes : "091" ne doit pas devenir 91.
    - is_missing_afm_hrs absent est déduit de afm_hrs (utile au §2.2 uniquement).
    - ev_season absente est déduite de ev_month (règle du NB01).

    Raises:
        ValueError: valeur non convertible dans le type attendu.
    """
    row: dict[str, Any] = {}

    for col in NUMERIC_INPUT_COLS:
        row[col] = _to_float(col, payload.get(col))

    for col in BOOL_INPUT_COLS:
        row[col] = _to_bool(payload.get(col))

    # Les espaces parasites sont retirés par engineer_features(), comme à l'entraînement.
    for col in CATEGORICAL_INPUT_COLS:
        value = payload.get(col)
        row[col] = np.nan if _is_missing(value) else str(value)

    if np.isnan(row["is_missing_afm_hrs"]):
        row["is_missing_afm_hrs"] = float(np.isnan(row["afm_hrs"]))

    if _is_missing(payload.get("ev_season")) and not np.isnan(row["ev_month"]):
        row["ev_season"] = SEASON_BY_MONTH.get(int(row["ev_month"]), np.nan)

    # Les drapeaux NOAA absents valent « non observé » (même règle que l'imputer
    # constant=0 du NB03) ; pilot_flying absent reste NaN → imputé (mode).
    for col in ("noaa_fog", "noaa_rain", "noaa_snow", "noaa_thunder"):
        if np.isnan(row[col]):
            row[col] = 0.0

    frame = pd.DataFrame([row], columns=RAW_INPUT_COLS)
    for col in CATEGORICAL_INPUT_COLS:
        frame[col] = frame[col].astype(object)
    return frame


def build_model_input(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Payload JSON → DataFrame (1 ligne × FEATURE_COLS) prêt pour preprocessing_pipeline.pkl."""
    return select_model_features(engineer_features(payload_to_frame(payload)))
