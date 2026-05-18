#!/usr/bin/env python
# coding: utf-8

# # 01 — Data Preparation
# **Aviation Safety Risk Prediction — NTSB + NOAA Dataset**
# 
# ### Changes vs v1
# - **Section order fixed**: sentinel/outlier fixes now happen *before* imputation (correct logical order)
# - **crew_age anomaly moved here (removed from NB02)**: values < 16 and > 100 are now set to NaN and median-imputed in this notebook
# - **Unit conversions and audit sections renumbered** for coherence
# 
# ### Pipeline order
# 1. Load & NOAA filter
# 2. Target inspection & leakage removal
# 3. Drop high-missing columns
# 4. NOAA sentinel → NaN
# 5. Physical outlier capping
# 6. `crew_age` anomaly fix ← **new position**
# 7. Shadow indicators (MNAR)
# 8. Temporal features
# 9. Drop useless columns
# 10. Imputation (median / mode / constant)
# 11. Post-audit fixes (celsius sentinels, additional outliers, near-constants, dedup)
# 12. Unit conversions (feet→m, miles→km, lbs→kg, gal→L, inHg→hPa)
# 13. Save clean dataset
# 

# In[1]:


import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import sys
print(sys.executable)
print(f"pandas {pd.__version__}  |  numpy {np.__version__}")


# ## SECTION 1 — Data Extraction from .mdb
# *(commented out — run once on Windows with Access driver)*

# In[2]:


# import pyodbc
# db_path = r"../data/raw/avall.mdb"
# tables_to_extract = ['aircraft', 'events', 'Flight_Crew', 'narratives']
# output_dir = '../data/ntsb_csv'
# os.makedirs(output_dir, exist_ok=True)
# conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};' + f'DBQ={os.path.abspath(db_path)};')
# conn = pyodbc.connect(conn_str)
# for table in tables_to_extract:
#     df = pd.read_sql(f"SELECT * FROM {table}", conn)
#     df.to_csv(os.path.join(output_dir, f'{table}.csv'), index=False, encoding='utf-8-sig')
# conn.close()


# ## SECTION 2 — Load CSVs & Merge NTSB Tables

# In[3]:


csv_dir = '../data/ntsb_csv'

events      = pd.read_csv(os.path.join(csv_dir, 'events.csv'),      dtype={'ev_id': str}, low_memory=False)
aircraft    = pd.read_csv(os.path.join(csv_dir, 'aircraft.csv'),    dtype={'ev_id': str}, low_memory=False)
flight_crew = pd.read_csv(os.path.join(csv_dir, 'Flight_Crew.csv'), dtype={'ev_id': str}, low_memory=False)
narratives  = pd.read_csv(os.path.join(csv_dir, 'narratives.csv'),  dtype={'ev_id': str}, low_memory=False)

print(f"events      : {events.shape}")
print(f"aircraft    : {aircraft.shape}")
print(f"flight_crew : {flight_crew.shape}")
print(f"narratives  : {narratives.shape}")


# In[4]:


# Merge: events + aircraft + crew + narratives
ntsb = events.merge(aircraft,    on='ev_id',                    how='left', suffixes=('', '_aircraft'))
ntsb = ntsb.merge(flight_crew,   on=['ev_id', 'Aircraft_Key'], how='left', suffixes=('', '_crew'))
ntsb = ntsb.merge(narratives,    on=['ev_id', 'Aircraft_Key'], how='left', suffixes=('', '_narr'))
print(f"Merged NTSB : {ntsb.shape}")


# ## SECTION 3 — NOAA Weather Enrichment
# *(commented out — run once, ~30 min)*

# In[5]:


# Full NOAA download + KD-Tree matching code omitted here for brevity.
# Output: ../data/processed/ntsb_noaa_merged.csv


# ## SECTION 4 — Fast Path: Load Already-Merged Data

# In[6]:


ntsb_enriched = pd.read_csv('../data/processed/ntsb_noaa_merged.csv', low_memory=False)
print(f"Loaded: {ntsb_enriched.shape}")


# In[7]:


# Keep only accidents with NOAA weather coverage
ntsb_final = ntsb_enriched[ntsb_enriched['noaa_temp_f'].notna()].copy()
print(f"With weather   : {len(ntsb_final):,}")
print(f"Without (dropped): {len(ntsb_enriched) - len(ntsb_final):,}")
print(f"Shape          : {ntsb_final.shape}")


# ## SECTION 5 — Missing Value Analysis

# In[8]:


missing = pd.DataFrame({
    'missing_count': ntsb_final.isnull().sum(),
    'missing_pct':   (ntsb_final.isnull().sum() / len(ntsb_final) * 100).round(2)
}).sort_values('missing_pct', ascending=False)

def tier(pct):
    if pct == 0:    return 'COMPLETE'
    elif pct >= 80: return 'DROP'
    elif pct >= 50: return 'REVIEW'
    elif pct >= 20: return 'IMPUTE_HIGH'
    else:           return 'IMPUTE_LOW'

missing['tier'] = missing['missing_pct'].apply(tier)
for t in ['COMPLETE', 'IMPUTE_LOW', 'IMPUTE_HIGH', 'REVIEW', 'DROP']:
    print(f"{t:12}: {(missing['tier'] == t).sum()} columns")


# ## SECTION 6 — Target Variable

# In[9]:


print(ntsb_final['ev_highest_injury'].value_counts(dropna=False))
ntsb_final = ntsb_final[ntsb_final['ev_highest_injury'].notna()].copy()
print(f"\nAfter dropping NaN targets: {ntsb_final.shape}")


# ## SECTION 7 — Remove Leakage Columns
# These columns are observed *after* the accident.  
# Using them to predict injury severity would be data leakage.
# 

# In[10]:


leakage_cols = [
    'inj_tot_f', 'inj_tot_m', 'inj_tot_n', 'inj_tot_s', 'inj_tot_t',
    'damage', 'acft_fire', 'acft_expl', 'crew_inj_level'
]
leakage_cols = [c for c in leakage_cols if c in ntsb_final.columns]
ntsb_final.drop(columns=leakage_cols, inplace=True)
print(f"Dropped {len(leakage_cols)} leakage columns — shape: {ntsb_final.shape}")


# ## SECTION 8 — Drop High-Missing Columns (>80%)

# In[11]:


drop_thresh = 0.80
cols_to_drop = [c for c in ntsb_final.columns if ntsb_final[c].isnull().mean() > drop_thresh]
ntsb_final.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} columns (>80% missing) — shape: {ntsb_final.shape}")


# ## SECTION 9 — Fix NOAA Sentinel Values
# NOAA uses `999.9` / `9999.9` to encode missing values. Replace with NaN **before any imputation**.
# 

# In[12]:


noaa_sentinels = {
    'noaa_dewp_f':        9999.9, 'noaa_slp_hpa':       9999.9,
    'noaa_visib_miles':   999.9,  'noaa_wind_knots':     999.9,
    'noaa_gust_knots':    999.9,  'noaa_snow_in':        999.9,
    'noaa_prcp_in':       99.99,  'noaa_maxwind_knots':  999.9,
    'noaa_temp_max_f':    9999.9, 'noaa_temp_min_f':     9999.9,
    'noaa_temp_max_c':    9999.9, 'noaa_temp_min_c':     9999.9,
}
for col, sentinel in noaa_sentinels.items():
    if col in ntsb_final.columns:
        n = (ntsb_final[col] == sentinel).sum()
        if n > 0:
            ntsb_final[col] = ntsb_final[col].replace(sentinel, np.nan)
            print(f"  {col}: {n:,} sentinels -> NaN")


# ## SECTION 10 — Physical Outlier Capping

# In[13]:


physical_caps = {
    'wind_dir_deg':   (0, 360),
    'wx_obs_dir':     (0, 360),
    'gust_kts':       (0, 150),
    'altimeter':      (27, 32),
    'wx_temp':        (-80, 130),
    'wx_dew_pt':      (-80, 100),
    'wx_obs_elev':    (-500, 15000),
    'vis_sm':         (0, 50),
    'afm_hrs':        (0, 50000),
    'cert_max_gr_wt': (0, 900000),
    'rwy_len':        (0, 20000),      # longest runway on Earth ~18,000 ft
    'rwy_width':      (0, 500),        # widest real runway ~300 ft
    'wind_vel_kts':   (0, 100),
    'wx_obs_dist':    (0, 500),
    'total_seats':    (0, 850),        # A380 max ~850
    'apt_elev':       (-500, 15000),
    'afm_hrs_last_insp': (0, 10000),
    'fuel_on_board':  (0, 50000),
}
for col, (lo, hi) in physical_caps.items():
    if col in ntsb_final.columns:
        n = ((ntsb_final[col] < lo) | (ntsb_final[col] > hi)).sum()
        if n > 0:
            ntsb_final.loc[(ntsb_final[col] < lo) | (ntsb_final[col] > hi), col] = np.nan
            print(f"  {col}: {n:,} out-of-range -> NaN")


# ## SECTION 11 — crew_age Anomaly Fix  ← **moved from NB02**
# Values < 16 are physiologically implausible for a pilot-in-command.  
# Values > 100 are almost certainly data entry errors.  
# We set them to NaN here and impute with the median in Section 13.  
# This removes the need to clamp in NB02.
# 

# In[14]:


if 'crew_age' in ntsb_final.columns:
    n_low  = (ntsb_final['crew_age'] < 16).sum()
    n_high = (ntsb_final['crew_age'] > 100).sum()
    print(f"  crew_age < 16 : {n_low:,} rows")
    print(f"  crew_age > 100: {n_high:,} rows")

    # Investigate distribution before fix
    print("\n  Distribution breakdown for age < 16:")
    if 'crew_category' in ntsb_final.columns:
        print(ntsb_final.loc[ntsb_final['crew_age'] < 16, 'crew_category'].value_counts())
    print("\n  Distribution breakdown for age > 100:")
    if 'crew_category' in ntsb_final.columns:
        print(ntsb_final.loc[ntsb_final['crew_age'] > 100, 'crew_category'].value_counts())

    # Set implausible values to NaN (imputed to median in Section 13)
    ntsb_final.loc[ntsb_final['crew_age'] < 16,  'crew_age'] = np.nan
    ntsb_final.loc[ntsb_final['crew_age'] > 100, 'crew_age'] = np.nan
    print(f"\n  crew_age anomalies set to NaN: {n_low + n_high:,} rows")
    print(f"  Remaining NaN crew_age: {ntsb_final['crew_age'].isnull().sum():,}")
    print(f"  New range: [{ntsb_final['crew_age'].min():.0f}, {ntsb_final['crew_age'].max():.0f}]")


# ## SECTION 12 — Shadow Indicators (MNAR Features)
# For columns with high missing rates, absence of data is itself informative.  
# Create binary `is_missing_X` flags **before** imputation.
# 

# In[15]:


shadow_cols = [
    'rwy_len', 'rwy_width', 'pax_seats', 'crew_age',
    'fuel_on_board', 'afm_hrs', 'afm_hrs_last_insp',
    'acft_year', 'wind_vel_kts',
]
for col in shadow_cols:
    if col in ntsb_final.columns:
        new_col = f'is_missing_{col}'
        ntsb_final[new_col] = ntsb_final[col].isnull().astype(int)
        pct = ntsb_final[new_col].mean() * 100
        print(f"  {new_col}: {ntsb_final[new_col].sum():,} flagged ({pct:.1f}%)")

print(f"\nShape after shadow indicators: {ntsb_final.shape}")


# ## SECTION 13 — Temporal Feature Engineering

# In[16]:


ntsb_final['ev_date'] = pd.to_datetime(ntsb_final['ev_date'], errors='coerce')
ntsb_final['ev_year']  = ntsb_final['ev_date'].dt.year.astype('Int64')
ntsb_final['ev_month'] = ntsb_final['ev_date'].dt.month.astype('Int64')
ntsb_final['ev_season'] = ntsb_final['ev_month'].map({
    12:'winter', 1:'winter',  2:'winter',
    3:'spring',  4:'spring',  5:'spring',
    6:'summer',  7:'summer',  8:'summer',
    9:'fall',   10:'fall',   11:'fall'
})
ntsb_final.drop(columns=['ev_date'], inplace=True)
print("Temporal features: ev_year, ev_month, ev_season")
print(f"Shape: {ntsb_final.shape}")


# ## SECTION 14 — Drop Useless Columns

# In[17]:


drop_useless = [
    'ev_id', 'ntsb_no', 'ntsb_no_aircraft', 'latitude', 'longitude',
    'lchg_date', 'lchg_date_aircraft', 'noaa_station_id', 'noaa_station_name',
    'commercial_space_flight', 'noaa_hail', 'noaa_tornado', 'acft_missing',
    'noaa_frshtt', 'ev_tmzn', 'lchg_date_crew', 'lchg_date_narr', 'date_last_insp',
    'narr_accf', 'narr_cause', 'narr_accp', 'metar',
    'ev_site_zipcode', 'wx_obs_fac_id', 'owner_zip', 'oper_zip',
    'dprt_apt_id', 'dprt_city', 'dprt_state', 'dprt_country',
    'dest_country', 'dest_city', 'dest_state',
    'crew_city', 'owner_city', 'oper_city', 'acft_serial_no', 'regis_no',
    'owner_state', 'oper_state', 'crew_res_state',
    'crew_res_country', 'oper_country', 'owner_country',
    'lchg_userid_crew', 'lchg_userid_narr', 'lchg_userid_aircraft', 'lchg_userid',
    'oper_street', 'owner_street', 'ft_as_of', 'bfr_date', 'date_lst_med',
    'dest_apt_id', 'apt_name', 'oper_name', 'certs_held', 'invest_agy',
    'noaa_snow_in', 'ev_city', 'ev_nr_apt_id', 'owner_acft', 'wx_obs_tmzn',
]
drop_useless = [c for c in drop_useless if c in ntsb_final.columns]
ntsb_final.drop(columns=drop_useless, inplace=True)
print(f"Dropped {len(drop_useless)} useless columns — shape: {ntsb_final.shape}")


# ## SECTION 15 — Imputation
# *(applied after outlier/anomaly fixes and shadow indicators)*

# In[18]:


# Numeric: median imputation
numeric_impute = [
    'ev_time', 'wx_obs_time', 'wx_obs_dir', 'wx_obs_elev',
    'vis_sm', 'altimeter', 'cert_max_gr_wt', 'total_seats',
    'num_eng', 'afm_hrs', 'crew_no',
    'noaa_dewp_f', 'noaa_visib_miles', 'noaa_wind_knots',
    'noaa_maxwind_knots', 'noaa_prcp_in', 'noaa_gust_knots',
    'noaa_slp_hpa', 'wx_dew_pt', 'wx_temp',
    'noaa_temp_max_f', 'noaa_temp_min_f', 'gust_kts', 'wind_dir_deg',
    'pax_seats', 'rwy_width', 'rwy_len', 'fc_seats',
    'apt_elev', 'acft_year', 'dprt_time', 'wind_vel_kts',
    'crew_age',              # ← anomalies already set to NaN in Section 11
    'afm_hrs_last_insp', 'fuel_on_board', 'apt_dir', 'apt_dist',
]
for col in numeric_impute:
    if col in ntsb_final.columns and ntsb_final[col].isnull().sum() > 0:
        ntsb_final[col] = ntsb_final[col].fillna(ntsb_final[col].median())

print("Numeric median imputation done")


# In[19]:


# Categorical: mode imputation
mode_impute = [
    'light_cond', 'wx_cond_basic', 'sky_cond_nonceil', 'sky_cond_ceil',
    'far_part', 'flt_plan_filed', 'type_fly', 'type_last_insp', 'elt_install',
    'crew_category', 'seat_occ_pic', 'pilot_flying', 'second_pilot', 'pc_profession',
    'ev_state', 'ev_nr_apt_loc', 'latlong_acq', 'wx_src_iic',
    'air_medical', 'site_seeing', 'rwy_num', 'elt_type', 'owner_acft_type',
    'infl_rest_inst', 'restraint_used', 'available_restraint',
    'crew_tox_perf', 'elt_oper', 'med_crtf_vldty',
    'flight_plan_activated', 'crew_sex', 'med_certf',
]
for col in mode_impute:
    if col in ntsb_final.columns and ntsb_final[col].isnull().sum() > 0:
        ntsb_final[col] = ntsb_final[col].fillna(ntsb_final[col].mode()[0])

print("Categorical mode imputation done")


# In[20]:


# Constant fills for high-cardinality / review-tier columns
constant_fills = {
    'acft_make':        'UNKNOWN', 'acft_model':       'UNKNOWN',
    'acft_category':    'UNKNOWN', 'acft_series':      'UNKNOWN',
    'elt_model':        'UNKNOWN', 'elt_manufacturer': 'UNKNOWN',
    'dprt_pt_same_ev':  'UNKNOWN', 'med_crtf_limit':   'NONE',
    'elt_aided_loc_ev': 'N',
}
for col, val in constant_fills.items():
    if col in ntsb_final.columns and ntsb_final[col].isnull().sum() > 0:
        ntsb_final[col] = ntsb_final[col].fillna(val)

print("Constant fill done")


# ## SECTION 16 — Post-Audit Fixes

# In[21]:


# Fix Celsius sentinels (values >80°C are physically impossible at Earth surface)
for col in ['noaa_temp_max_c', 'noaa_temp_min_c']:
    if col in ntsb_final.columns:
        n = (ntsb_final[col] > 80).sum()
        ntsb_final.loc[ntsb_final[col] > 80, col] = np.nan
        ntsb_final[col] = ntsb_final[col].fillna(ntsb_final[col].median())
        print(f"  {col}: {n} values >80°C fixed")

# acft_year: Wright Brothers flew 1903 — nothing earlier is valid
if 'acft_year' in ntsb_final.columns:
    n_zero = (ntsb_final['acft_year'] == 0).sum()
    n_old  = (ntsb_final['acft_year'] < 1903).sum()
    ntsb_final.loc[ntsb_final['acft_year'] <= 0, 'acft_year'] = np.nan
    ntsb_final.loc[ntsb_final['acft_year'] < 1903, 'acft_year'] = np.nan
    ntsb_final['acft_year'] = ntsb_final['acft_year'].fillna(ntsb_final['acft_year'].median())
    print(f"  acft_year: {n_zero} zeros and {n_old} <1903 fixed")

# Drop near-constant columns (no learning signal)
drop_constants = [
    'ev_type',         # 97.2% ACC
    'ev_country',      # 99.9% USA
    'elt_aided_loc_ev',# 97.3% N
    'site_seeing',     # 96.0% N
    'air_medical',     # 98.9% N
    'unmanned',        # 99.8% False
]
drop_constants = [c for c in drop_constants if c in ntsb_final.columns]
ntsb_final.drop(columns=drop_constants, inplace=True)
print(f"  Dropped {len(drop_constants)} near-constant columns")

# Deduplicate (one row per accident — pilot in command only)
before = len(ntsb_final)
ntsb_final = ntsb_final.drop_duplicates(keep='first').reset_index(drop=True)
after = len(ntsb_final)
print(f"  Duplicates removed: {before - after:,} ({before:,} -> {after:,})")
print(f"  Shape: {ntsb_final.shape}")


# ## SECTION 17 — Unit Consistency: Convert to SI

# In[22]:


# wx_temp / wx_dew_pt: Fahrenheit → Celsius
ntsb_final['wx_temp']   = (ntsb_final['wx_temp']   - 32) * 5/9
ntsb_final['wx_dew_pt'] = (ntsb_final['wx_dew_pt'] - 32) * 5/9
ntsb_final.rename(columns={'wx_temp': 'wx_temp_c', 'wx_dew_pt': 'wx_dew_pt_c'}, inplace=True)
drop_f_cols = [c for c in ['noaa_temp_f', 'noaa_dewp_f', 'noaa_temp_max_f', 'noaa_temp_min_f']
               if c in ntsb_final.columns]
ntsb_final.drop(columns=drop_f_cols, inplace=True)
print(f"Dropped {len(drop_f_cols)} Fahrenheit duplicates")


# In[23]:


# FEET -> METERS
feet_to_m = 0.3048
for old, new in {'apt_elev':'apt_elev_m','wx_obs_elev':'wx_obs_elev_m',
                  'sky_ceil_ht':'sky_ceil_ht_m','sky_nonceil_ht':'sky_nonceil_ht_m',
                  'rwy_len':'rwy_len_m','rwy_width':'rwy_width_m'}.items():
    if old in ntsb_final.columns:
        ntsb_final[new] = (ntsb_final[old] * feet_to_m).round(1)
        ntsb_final.drop(columns=[old], inplace=True)
        print(f"  {old:22} -> {new}")

# STATUTE MILES -> KILOMETERS
for old, new in {'vis_sm':'vis_km','noaa_visib_miles':'noaa_visib_km',
                  'apt_dist':'apt_dist_km','wx_obs_dist':'wx_obs_dist_km'}.items():
    if old in ntsb_final.columns:
        ntsb_final[new] = (ntsb_final[old] * 1.60934).round(3)
        ntsb_final.drop(columns=[old], inplace=True)
        print(f"  {old:22} -> {new}")

# POUNDS -> KILOGRAMS
if 'cert_max_gr_wt' in ntsb_final.columns:
    ntsb_final['cert_max_gr_wt_kg'] = (ntsb_final['cert_max_gr_wt'] * 0.453592).round(1)
    ntsb_final.drop(columns=['cert_max_gr_wt'], inplace=True)
    print("  cert_max_gr_wt        -> cert_max_gr_wt_kg")

# GALLONS -> LITERS
if 'fuel_on_board' in ntsb_final.columns:
    ntsb_final['fuel_on_board_l'] = (ntsb_final['fuel_on_board'] * 3.78541).round(1)
    ntsb_final.drop(columns=['fuel_on_board'], inplace=True)
    print("  fuel_on_board         -> fuel_on_board_l")

# INCHES OF MERCURY -> hPa
if 'altimeter' in ntsb_final.columns:
    ntsb_final['altimeter_hpa'] = (ntsb_final['altimeter'] * 33.8639).round(2)
    ntsb_final.drop(columns=['altimeter'], inplace=True)
    print("  altimeter             -> altimeter_hpa")

# INCHES -> MILLIMETERS (precipitation)
if 'noaa_prcp_in' in ntsb_final.columns:
    ntsb_final['noaa_prcp_mm'] = (ntsb_final['noaa_prcp_in'] * 25.4).round(2)
    ntsb_final.drop(columns=['noaa_prcp_in'], inplace=True)
    print("  noaa_prcp_in          -> noaa_prcp_mm")

# Knots kept as-is (international aviation standard)
print(f"\nFinal shape : {ntsb_final.shape}")
print(f"Total NaN   : {ntsb_final.isnull().sum().sum()}")


# ## SECTION 18 — Final Verification

# In[24]:


total_nan = ntsb_final.isnull().sum().sum()
remaining = ntsb_final.isnull().sum()
remaining = remaining[remaining > 0].sort_values(ascending=False)

print(f"Shape        : {ntsb_final.shape}")
print(f"Total NaN    : {total_nan}")
print(f"\nTarget distribution:")
print(ntsb_final['ev_highest_injury'].value_counts())

if len(remaining) > 0:
    print(f"\nColumns still with NaN (investigate before proceeding):")
    print(remaining)
else:
    print("\nAll columns clean — zero NaN ✓")

# Verify shadow indicators
shadow_created = [c for c in ntsb_final.columns if c.startswith('is_missing_')]
print(f"\nShadow indicators: {len(shadow_created)}")
for col in shadow_created:
    print(f"  {col}: {ntsb_final[col].sum():,} flagged ({ntsb_final[col].mean()*100:.1f}%)")

# Verify crew_age range (should be [16, 100])
print(f"\ncrew_age range: [{ntsb_final['crew_age'].min():.0f}, {ntsb_final['crew_age'].max():.0f}]")
print(f"  Values < 16  : {(ntsb_final['crew_age'] < 16).sum()}  (should be 0)")
print(f"  Values > 100 : {(ntsb_final['crew_age'] > 100).sum()} (should be 0)")


# ## SECTION 19 — Save Clean Dataset

# In[25]:


output_path = '../data/processed/ntsb_clean_final.csv'
ntsb_final.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Shape: {ntsb_final.shape}")
print(f"Columns: {ntsb_final.shape[1]}")

