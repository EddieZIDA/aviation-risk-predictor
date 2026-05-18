"""
utils/feature_engineering.py — Feature Engineering Pipeline
==========================================================
Reproduit fidèlement les transformations de notebooks/03_feature_engineering.ipynb
pour aligner le backend avec le pipeline ML.

Flux :
  1. Conversions unités impériales → SI (°F→°C, ft→m, in→mm, etc.)
  2. Corrections d'anomalies (afm_hrs==0, gust sans wind)
  3. Features dérivées (cyclical, ratios, flags, etc.)
  4. Retourne le DataFrame avec FEATURE_COLS prêtes pour le pipeline sklearn
"""

import numpy as np
import pandas as pd
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# ── Colonnes brutes attendues du frontend (NTSB) ────────────────────────────────
RAW_INPUT_COLS: List[str] = [
    # Événement
    'ev_state', 'ev_year', 'ev_month', 'ev_dow', 'ev_time',
    
    # Météo NTSB (unités impériales)
    'wx_temp', 'wx_dew_pt', 'altimeter', 'vis_sm', 'wind_dir_deg', 'wind_vel_kts', 
    'gust_kts', 'sky_ceil_ht', 'sky_nonceil_ht', 'light_cond', 'sky_cond_nonceil', 
    'sky_cond_ceil', 'wx_cond_basic', 'wx_src_iic',
    
    # NOAA (déjà SI)
    'noaa_temp_c', 'noaa_wind_knots', 'noaa_gust_knots', 'noaa_visib_km', 
    'noaa_slp_hpa', 'noaa_prcp_in', 'noaa_fog', 'noaa_rain', 'noaa_snow', 
    'noaa_thunder', 'noaa_dist_km',
    
    # Appareil
    'acft_make', 'acft_category', 'acft_year', 'fixed_retractable', 'num_eng', 
    'cert_max_gr_wt', 'fuel_on_board', 'homebuilt',
    
    # Heures de cellule
    'afm_hrs', 'afm_hrs_last_insp',
    
    # Piste / aéroport
    'rwy_len', 'rwy_width', 'apt_elev', 'apt_dist', 'pax_seats', 'total_seats',
    
    # Opérations
    'far_part', 'type_fly', 'flt_plan_filed',
    
    # Équipage
    'crew_age', 'crew_sex', 'crew_category', 'pilot_flying', 'second_pilot', 
    'med_certf', 'med_crtf_vldty', 'pc_profession',
]

# ── Colonnes finales attendues par le pipeline ML (après FE) ───────────────────
# Ces colonnes correspondent exactement à FEATURE_COLS du notebook 03
FEATURE_COLS: List[str] = [
    # Colonnes binaires Y/N
    'homebuilt', 'second_pilot', 'afm_hrs_since', 'elt_install', 'elt_oper',
    'crew_tox_perf', 'latlong_acq', 'ev_nr_apt_loc', 'wind_dir_ind', 'wind_vel_ind',
    'gust_ind', 'pilot_flying',
    
    # Colonnes binaires numériques
    'noaa_fog', 'noaa_rain', 'noaa_snow', 'noaa_thunder', 'adverse_weather',
    'poor_visibility', 'high_wind', 'extreme_temp', 'fog_risk', 'is_imc',
    'is_missing_rwy_len', 'is_missing_rwy_width', 'is_missing_pax_seats',
    'is_missing_crew_age', 'is_missing_fuel_on_board', 'is_missing_afm_hrs',
    'is_missing_afm_hrs_last_insp', 'is_missing_acft_year', 'is_missing_wind_vel_kts',
    
    # Colonnes One-Hot encodées (base)
    'acft_category', 'ev_season', 'light_cond', 'sky_cond_nonceil', 'sky_cond_ceil',
    'wx_src_iic', 'type_last_insp', 'crew_category', 'med_certf', 'med_crtf_vldty',
    'fixed_retractable', 'acft_make_grouped', 'seat_occ_pic',
    
    # Colonnes One-Hot encodées (haute cardinalité)
    'ev_state', 'far_part', 'type_fly', 'flt_plan_filed', 'flight_plan_activated',
    'pc_profession', 'infl_rest_inst', 'dprt_pt_same_ev', 'elt_type',
    'available_restraint', 'restraint_used', 'med_crtf_limit',
    
    # Colonnes ordinales
    'crew_sex',
    
    # Colonnes numériques
    'noaa_temp_c', 'noaa_temp_max_c', 'noaa_temp_min_c', 'noaa_wind_knots',
    'noaa_maxwind_knots', 'noaa_gust_knots', 'noaa_slp_hpa', 'noaa_visib_km',
    'noaa_prcp_mm', 'noaa_dist_km', 'gust_excess', 'wx_temp_c', 'wx_dew_pt_c',
    'wind_dir_deg', 'wind_vel_kts', 'gust_kts', 'vis_km', 'sky_ceil_ht_m',
    'sky_nonceil_ht_m', 'altimeter_hpa', 'td_spread', 'apt_elev_m', 'apt_dist_km',
    'wx_obs_elev_m', 'wx_obs_dist_km', 'rwy_width_m', 'rwy_len_m_log', 'num_eng',
    'fc_seats', 'pax_seats', 'total_seats', 'afm_hrs_log', 'afm_hrs_last_insp',
    'cert_max_gr_wt_kg_log', 'fuel_on_board_l_log', 'acft_age', 'maintenance_ratio',
    'crew_age', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'ev_year', 'wx_obs_time',
]

def _sanitize_lgb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes pour LightGBM (caractères spéciaux → _)."""
    df = df.copy()
    new_cols = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
    seen = {}
    deduped = []
    for col in new_cols:
        if col in seen:
            seen[col] += 1
            deduped.append(f'{col}_{seen[col]}')
        else:
            seen[col] = 0
            deduped.append(col)
    df.columns = deduped
    return df

def apply_feature_engineering(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le feature engineering complet sur les données brutes NTSB.
    
    Args:
        raw_df: DataFrame avec les colonnes brutes RAW_INPUT_COLS
        
    Returns:
        DataFrame avec les colonnes FEATURE_COLS prêt pour le pipeline ML
    """
    df = raw_df.copy()
    
    # 1. Conversions unités impériales → SI
    # Température: °F → °C
    df['wx_temp_c'] = (df['wx_temp'] - 32) * 5/9
    df['wx_dew_pt_c'] = (df['wx_dew_pt'] - 32) * 5/9
    
    # Pression: inHg → hPa
    df['altimeter_hpa'] = df['altimeter'] * 33.8639
    
    # Distance: statute miles → km
    df['vis_km'] = df['vis_sm'] * 1.60934
    
    # Hauteur: pieds → mètres
    df['sky_ceil_ht_m'] = df['sky_ceil_ht'] * 0.3048
    df['sky_nonceil_ht_m'] = df['sky_nonceil_ht'] * 0.3048
    
    # Précipitation: pouces → mm
    df['noaa_prcp_mm'] = df['noaa_prcp_in'] * 25.4
    
    # Poids: livres → kg
    df['cert_max_gr_wt_kg'] = df['cert_max_gr_wt'] * 0.453592
    
    # Carburant: gallons → litres
    df['fuel_on_board_l'] = df['fuel_on_board'] * 3.78541
    
    # Piste: pieds → mètres
    df['rwy_len_m'] = df['rwy_len'] * 0.3048
    df['rwy_width_m'] = df['rwy_width'] * 0.3048
    
    # Altitude: pieds → mètres
    df['apt_elev_m'] = df['apt_elev'] * 0.3048
    
    # Distance: miles → km
    df['apt_dist_km'] = df['apt_dist'] * 1.60934
    
    # 2. Corrections d'anomalies
    # afm_hrs == 0 → NaN (sauf si flag missing)
    mask_zero_hrs = (df['afm_hrs'] == 0) & (~df.get('is_missing_afm_hrs', pd.Series(False, index=df.index)).astype(bool))
    df.loc[mask_zero_hrs, 'afm_hrs'] = np.nan
    
    # Gust avec wind = 0
    mask_gust_no_wind = (df['noaa_gust_knots'].notna()) & (df['noaa_wind_knots'] == 0)
    df.loc[mask_gust_no_wind, 'noaa_wind_knots'] = df.loc[mask_gust_no_wind, 'noaa_gust_knots']
    
    # 3. Feature engineering dérivé
    # Flags météo
    df['adverse_weather'] = (
        df['noaa_fog'].astype(bool) | df['noaa_rain'].astype(bool) |
        df['noaa_snow'].astype(bool) | df['noaa_thunder'].astype(bool)
    ).astype(int)
    
    df['poor_visibility'] = (df['vis_km'] < 5).astype(int)
    df['high_wind'] = (df['noaa_wind_knots'] > 25).astype(int)
    df['extreme_temp'] = ((df['noaa_temp_c'] < -20) | (df['noaa_temp_c'] > 35)).astype(int)
    df['td_spread'] = df['wx_temp_c'] - df['wx_dew_pt_c']
    df['fog_risk'] = (df['td_spread'] < 3).astype(int)
    df['is_imc'] = (df['wx_cond_basic'] == 'IMC').astype(int)
    
    # Maintenance ratio
    eps = 1e-3
    df['maintenance_ratio'] = (
        (df['afm_hrs'] - df['afm_hrs_last_insp']).clip(lower=0) / (df['afm_hrs'] + eps)
    ).clip(0, 1)
    
    # Features temporelles cycliques
    df['ev_hour'] = (df['ev_time'] // 100).clip(0, 23)
    df['hour_sin'] = np.sin(2 * np.pi * df['ev_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['ev_hour'] / 24)
    
    dow_map = {'Mo':0, 'Tu':1, 'We':2, 'Th':3, 'Fr':4, 'Sa':5, 'Su':6}
    df['ev_dow_num'] = df['ev_dow'].map(dow_map)
    df['dow_sin'] = np.sin(2 * np.pi * df['ev_dow_num'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['ev_dow_num'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['ev_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['ev_month'] / 12)
    
    # Aircraft age
    df['acft_age'] = (df['ev_year'] - df['acft_year']).clip(lower=0, upper=80)
    
    # Gust excess
    df['gust_excess'] = (df['noaa_gust_knots'] - df['noaa_wind_knots']).clip(lower=0).fillna(0)
    
    # Log transforms
    log_cols = ['afm_hrs', 'cert_max_gr_wt_kg', 'fuel_on_board_l', 'rwy_len_m']
    for col in log_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    # 4. Groupement des fabricants (top 25 + OTHER)
    if 'acft_make' in df.columns:
        df['acft_make'] = df['acft_make'].str.upper().str.strip()
        make_aliases = {
            'ROBINSON HELICOPTER COMPANY': 'ROBINSON', 'ROBINSON HELICOPTER': 'ROBINSON',
            'CIRRUS DESIGN CORP': 'CIRRUS', 'CIRRUS DESIGN': 'CIRRUS',
            'DIAMOND AIRCRAFT IND INC': 'DIAMOND', 'DIAMOND AIRCRAFT': 'DIAMOND',
            'AIR TRACTOR INC': 'AIR TRACTOR', 'BOMBARDIER INC': 'BOMBARDIER',
            'DEHAVILLAND': 'DE HAVILLAND', 'DE HAVILLAND CANADA': 'DE HAVILLAND',
        }
        df['acft_make'] = df['acft_make'].replace(make_aliases)
        
        # Top 25 makes + OTHER
        top_25_makes = ['CESSNA', 'PIPER', 'BEECH', 'ROBINSON', 'BOEING', 'BELL', 
                       'CIRRUS', 'AIR TRACTOR', 'MOONEY', 'AIRBUS', 'SCHWEIZER', 
                       'BELLANCA', 'DE HAVILLAND', 'AERONCA', 'MAULE', 'DIAMOND', 
                       'HUGHES', 'BOMBARDIER', 'VANS', 'CHAMPION', 'LUSCOMBE', 
                       'EMBRAER', 'STINSON', 'EUROCOPTER', 'NORTH AMERICAN']
        df['acft_make_grouped'] = df['acft_make'].apply(
            lambda x: x if x in top_25_makes else 'OTHER'
        )
    
    # 5. Flags de valeurs manquantes (pour compatibilité avec le pipeline)
    missing_cols = [
        'rwy_len', 'rwy_width', 'pax_seats', 'crew_age', 'fuel_on_board',
        'afm_hrs', 'afm_hrs_last_insp', 'acft_year', 'wind_vel_kts'
    ]
    for col in missing_cols:
        if col in df.columns:
            df[f'is_missing_{col}'] = df[col].isna().astype(int)
    
    # 6. S'assurer que toutes les colonnes FEATURE_COLS sont présentes
    # Colonnes manquantes → NaN
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    
    # 7. Retourner uniquement les colonnes attendues par le pipeline
    result_df = df[FEATURE_COLS].copy()
    
    # S'assurer qu'on a bien 416 colonnes
    if len(result_df.columns) != 416:
        logger.warning(f"Attention : {len(result_df.columns)} features générées, 416 attendues")
    
    # Nettoyage final pour LightGBM (si nécessaire)
    # result_df = _sanitize_lgb_columns(result_df)
    
    return result_df
