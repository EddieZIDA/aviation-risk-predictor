# 05 — Quantitative Uncertainty — Conformal Prediction
# Aviation Safety Risk Prediction — NTSB + NOAA Dataset

import os
import sys
import re
import warnings
from pathlib import Path

# Installation silencieuse de mapie si absent
import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', 'mapie', '-q'], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import mapie
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score, classification_mean_width_score
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# ─── 1. CONFIGURATION ET CHEMINS ──────────────────────────────────────────────
# On s'assure d'être dans le dossier "notebooks" pour que les chemins relatifs fonctionnent
os.chdir(Path(__file__).resolve().parent)

SEED       = 42
OUTPUT_DIR = 'outputs'
MODEL_DIR  = 'baseline_models'
UNCERT_DIR = 'uncertainty_outputs'
os.makedirs(UNCERT_DIR, exist_ok=True)

LABEL_MAP   = {0:'NONE', 1:'MINR', 2:'SERS', 3:'FATL'}
LABEL_NAMES = ['NONE', 'MINR', 'SERS', 'FATL']
CLASS_ORDER = [0, 1, 2, 3]
ALPHA_LEVELS = [0.05, 0.10, 0.15, 0.20]
ALPHA_MAIN = 0.10

np.random.seed(SEED)
print(f'MAPIE version: {mapie.__version__}')
print('Imports et configuration terminés ✓')

# ─── 2. CHARGEMENT DU MODÈLE ET DES DONNÉES ───────────────────────────────────
best_model = joblib.load(f'{MODEL_DIR}/best_model.pkl')
print(f'Best model type: {type(best_model).__name__}')

X_val_raw  = pd.read_csv(f'{OUTPUT_DIR}/X_val.csv')
X_test_raw = pd.read_csv(f'{OUTPUT_DIR}/X_test.csv')
y_val  = pd.read_csv(f'{OUTPUT_DIR}/y_val.csv').squeeze()
y_test = pd.read_csv(f'{OUTPUT_DIR}/y_test.csv').squeeze()

def sanitize_lgb_columns(df):
    df = df.copy()
    new_cols = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
    seen = {}; deduped = []
    for col in new_cols:
        if col in seen: seen[col] += 1; deduped.append(f'{col}_{seen[col]}')
        else: seen[col] = 0; deduped.append(col)
    df.columns = deduped
    return df

is_lgb = 'LGBM' in type(best_model).__name__
if is_lgb:
    X_val  = sanitize_lgb_columns(X_val_raw)
    X_test = sanitize_lgb_columns(X_test_raw)
    print('LightGBM détecté — colonnes nettoyées')
else:
    X_val  = X_val_raw.values
    X_test = X_test_raw.values

print(f'X_val  shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')

# ─── 3. CALIBRATION CONFORMAL PREDICTION ──────────────────────────────────────
mapie = MapieClassifier(
    estimator=best_model,
    method='lac',
    cv='prefit',
    random_state=SEED,
)
mapie.fit(X_val, y_val)
mapie_90 = mapie
print('LAC calibré sur le set de validation ✓')

# ─── 4. ÉVALUATION DE LA COUVERTURE (COVERAGE) ────────────────────────────────
coverage_results = []

for alpha in ALPHA_LEVELS:
    m = MapieClassifier(
        estimator=best_model,
        method='lac',
        cv='prefit',
        random_state=SEED,
    )
    m.fit(X_val, y_val)
    
    # Remplacement de predict_set par predict (Nouvelle API Mapie)
    _, pred_sets_raw = m.predict(X_test, alpha=alpha)
    pred_sets = pred_sets_raw[:, :, 0]

    empirical_coverage = classification_coverage_score(y_test.values, pred_sets)
    mean_set_size      = classification_mean_width_score(pred_sets)
    set_sizes          = pred_sets.sum(axis=1)
    frac_singleton     = (set_sizes == 1).mean()
    frac_full_set      = (set_sizes == 4).mean()

    coverage_results.append({
        'alpha': alpha,
        'target_coverage': 1 - alpha,
        'empirical_coverage': float(empirical_coverage),
        'mean_set_size':      float(mean_set_size),
        'frac_singleton':     float(frac_singleton),
        'frac_full_set':      float(frac_full_set),
    })
    print(f"  α={alpha:.2f}  target={(1-alpha)*100:.0f}%  "
          f"empirical={float(empirical_coverage):.3f}  "
          f"mean_set_size={float(mean_set_size):.2f}  "
          f"singletons={float(frac_singleton)*100:.1f}%")

cov_df = pd.DataFrame(coverage_results)
cov_df.to_csv(f'{UNCERT_DIR}/coverage_results.csv', index=False)

# ─── 5. ANALYSE À ALPHA = 0.10 ────────────────────────────────────────────────
y_pred_point_main, pred_sets_raw = mapie_90.predict(X_test, alpha=ALPHA_MAIN)
pred_sets_main = pred_sets_raw[:, :, 0]
set_sizes_main = pred_sets_main.sum(axis=1)

vc_size = pd.Series(set_sizes_main).value_counts().sort_index()
print(f'\nAt α={ALPHA_MAIN} (90% coverage):')
for size, count in vc_size.items():
    print(f'  Size {size}: {count:,} samples ({count/len(y_test)*100:.1f}%)')

# ─── 6. SAUVEGARDE DU MODÈLE ET DES ARTEFACTS ─────────────────────────────────
# Sauvegarde du modèle MAPIE pour l'API Flask
joblib.dump(mapie, f'{UNCERT_DIR}/mapie_classifier_lac.pkl')
print(f'\nLAC model sauvegardé avec succès dans : {UNCERT_DIR}/mapie_classifier_lac.pkl')

print('\n' + '='*65)
print('  CONFORMAL PREDICTION — SUMMARY REPORT')
print('='*65)
best_row = cov_df[cov_df['alpha'] == 0.10].iloc[0]
print(f'  Base model               : {type(best_model).__name__}')
print(f'  At α=0.10 (90% target coverage):')
print(f'    Empirical coverage     : {best_row["empirical_coverage"]:.4f}')
print(f'    Mean set size          : {best_row["mean_set_size"]:.3f}')
print(f'  Coverage guarantee holds : {best_row["empirical_coverage"] >= 1 - 0.10}')
print('='*65)
print("FIN DU SCRIPT. L'API PEUT MAINTENANT UTILISER L'INCERTITUDE.")