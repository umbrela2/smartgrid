"""
solar_config.py - Configuration partagée de la partie B (solaire + LSTM).

Tous les paramètres modifiables sont ici : coordonnées, période, hypothèses
photovoltaïques, hyper-paramètres du LSTM. Ne rien "coder en dur" ailleurs.
"""
from pathlib import Path

# --- Chemins (relatifs à la racine du dépôt) ---------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

# --- Localisation et période (alignées sur le dataset Smart Meters London) ---
LAT, LON = 51.5074, -0.1278          # Londres
START, END = "2011-11-23", "2014-02-28"

# --- Hypothèses de la production PV simulée ----------------------------------
# pv_kwh = PV_SHARE * PV_SURFACE_M2 * PV_EFFICIENCY * PV_PR * irradiance/1000
# => production MOYENNE PAR FOYER (tous foyers confondus, équipés ou non),
#    exprimée en kWh sur 1 heure (= kW moyen).
PV_SURFACE_M2 = 10.0     # surface de panneaux par foyer équipé (m²)
PV_EFFICIENCY = 0.18     # rendement des panneaux
PV_PR = 0.80             # performance ratio (pertes onduleur, câbles, poussière...)
PV_SHARE = 0.25          # part des foyers équipés de panneaux

# --- LSTM --------------------------------------------------------------------
SEED = 42
LOOKBACK = 24            # heures passées en entrée
HORIZON = 6              # heures futures prédites (t+1 ... t+6)
FEATURES = ["irradiance", "temp", "cloud",
            "hour_sin", "hour_cos", "month_sin", "month_cos"]
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15    # le reste (15 %) = test, découpage chronologique
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
DAY_THRESHOLD_WM2 = 5.0  # irradiance > seuil => "heure de jour" (évaluation hors nuit)


def ensure_dirs():
    for p in (DATA_RAW, DATA_PROC, MODELS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
