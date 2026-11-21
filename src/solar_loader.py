"""
solar_loader.py - Téléchargement de l'irradiance (Open-Meteo) + calcul de la
production photovoltaïque simulée.

Usage :
    python src/solar_loader.py              # données réelles (Internet requis)
    python src/solar_loader.py --synthetic  # données FICTIVES (test/secours uniquement)

Sorties :
    data/raw/solar_london.csv        données brutes de l'API
    data/processed/solar_pv.csv      timestamp, irradiance, temp, cloud, pv_kwh
"""
import argparse
import time

import numpy as np
import pandas as pd
import requests

from solar_config import (DATA_PROC, DATA_RAW, END, LAT, LON, PV_EFFICIENCY, PV_PR,
                          PV_SHARE, PV_SURFACE_M2, SEED, START, ensure_dirs)

API_URL = "https://archive-api.open-meteo.com/v1/archive"
RENAME = {"time": "timestamp", "shortwave_radiation": "irradiance",
          "temperature_2m": "temp", "cloud_cover": "cloud"}


def fetch_chunk(start: str, end: str, retries: int = 4) -> pd.DataFrame:
    """Télécharge un morceau [start, end] (par année pour rester sous les limites de l'API)."""
    params = {"latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
              "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
              "timezone": "UTC"}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=90)
            if r.status_code == 200:
                return pd.DataFrame(r.json()["hourly"])
            print(f"  [!] HTTP {r.status_code} pour {start}->{end} (essai {attempt}/{retries}): {r.text[:150]}")
        except requests.RequestException as exc:
            print(f"  [!] Erreur réseau (essai {attempt}/{retries}): {exc}")
        time.sleep(10 * attempt)
    raise RuntimeError(f"Échec du téléchargement {start}->{end}. Voir le guide, section Dépannage.")


def download_real() -> pd.DataFrame:
    s, e = pd.Timestamp(START), pd.Timestamp(END)
    parts = []
    for year in range(s.year, e.year + 1):
        a = max(s, pd.Timestamp(year=year, month=1, day=1))
        b = min(e, pd.Timestamp(year=year, month=12, day=31))
        print(f"Téléchargement {a.date()} -> {b.date()} ...")
        parts.append(fetch_chunk(str(a.date()), str(b.date())))
        time.sleep(2)
    return pd.concat(parts, ignore_index=True)


def make_synthetic() -> pd.DataFrame:
    """Irradiance FICTIVE (géométrie solaire + nuages aléatoires). Pour tester le code sans Internet."""
    rng = np.random.default_rng(SEED)
    idx = pd.date_range(START, END + " 23:00", freq="h")
    doy, hour = idx.dayofyear.values, idx.hour.values
    decl = np.radians(23.45 * np.sin(2 * np.pi * (284 + doy) / 365))
    lat = np.radians(LAT)
    ha = np.radians(15 * (hour - 12))
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    clear = np.clip(cosz, 0, None) * 950
    x, cloud = 0.0, np.empty(len(idx))
    for t in range(len(idx)):
        x = 0.97 * x + rng.normal(0, 0.3)
        cloud[t] = 100 / (1 + np.exp(-1.5 * x))
    irr = clear * (1 - 0.75 * (cloud / 100) ** 1.5) * rng.uniform(0.95, 1.05, len(idx))
    temp = 10 + 8 * np.sin(2 * np.pi * (doy - 110) / 365) + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1, len(idx))
    return pd.DataFrame({"time": idx.strftime("%Y-%m-%dT%H:%M"), "shortwave_radiation": irr,
                         "temperature_2m": temp, "cloud_cover": cloud})


def clean_and_compute_pv(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=RENAME).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    n_missing = len(full) - len(df)
    df = df.reindex(full)
    df.index.name = "timestamp"
    n_nan = int(df.isna().any(axis=1).sum())
    df = df.interpolate(limit=6).ffill().bfill()
    df["irradiance"] = df["irradiance"].clip(lower=0)
    df["cloud"] = df["cloud"].clip(0, 100)
    # Production PV moyenne par foyer (kWh sur 1 h) :
    df["pv_kwh"] = PV_SHARE * PV_SURFACE_M2 * PV_EFFICIENCY * PV_PR * df["irradiance"] / 1000.0
    print(f"Lignes : {len(df)} | heures manquantes comblées : {n_missing} | lignes avec NaN interpolées : {n_nan}")
    return df.reset_index()[["timestamp", "irradiance", "temp", "cloud", "pv_kwh"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="génère des données fictives (NE PAS utiliser pour le rapport final)")
    args = ap.parse_args()
    ensure_dirs()
    if args.synthetic:
        print("\n" + "!" * 70 + "\n DONNÉES FICTIVES : uniquement pour tester le code !\n" + "!" * 70 + "\n")
        raw = make_synthetic()
        raw.to_csv(DATA_RAW / "solar_synthetic.csv", index=False)
    else:
        raw = download_real()
        raw.to_csv(DATA_RAW / "solar_london.csv", index=False)
    df = clean_and_compute_pv(raw)
    df.to_csv(DATA_PROC / "solar_pv.csv", index=False)
    print(f"OK -> {DATA_PROC / 'solar_pv.csv'}")
    print(df.describe().round(3).to_string())


if __name__ == "__main__":
    main()
