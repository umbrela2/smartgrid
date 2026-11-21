"""
solar_eda.py - Analyse exploratoire des données solaires (figures pour le rapport).
Usage : python src/solar_eda.py
Sorties : reports/figures/eda_*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solar_config import DATA_PROC, FIGURES, ensure_dirs


def main():
    ensure_dirs()
    df = pd.read_csv(DATA_PROC / "solar_pv.csv", parse_dates=["timestamp"]).set_index("timestamp")
    df["hour"], df["month"] = df.index.hour, df.index.month

    # 1. Profil journalier moyen par saison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    seasons = {"Hiver (D-J-F)": [12, 1, 2], "Printemps (M-A-M)": [3, 4, 5],
               "Été (J-J-A)": [6, 7, 8], "Automne (S-O-N)": [9, 10, 11]}
    for name, months in seasons.items():
        s = df[df["month"].isin(months)].groupby("hour")["pv_kwh"].mean()
        ax.plot(s.index, s.values, label=name, marker="o", ms=3)
    ax.set(xlabel="Heure (UTC)", ylabel="Production PV moyenne (kWh/foyer/h)", title="Profil journalier de production PV")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGURES / "eda_profil_journalier.png", dpi=150); plt.close(fig)

    # 2. Production mensuelle totale
    m = df.groupby(df.index.to_period("M"))["pv_kwh"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(m.index.astype(str), m.values); ax.set_xticks(range(0, len(m), 3))
    ax.set_xticklabels(m.index.astype(str)[::3], rotation=45)
    ax.set(ylabel="kWh/foyer/mois", title="Production PV mensuelle (saisonnalité)")
    fig.tight_layout(); fig.savefig(FIGURES / "eda_mensuel.png", dpi=150); plt.close(fig)

    # 3. Heatmap heure x mois
    pivot = df.pivot_table(index="hour", columns="month", values="irradiance", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="YlOrRd")
    ax.set(xticks=range(12), xticklabels=range(1, 13), xlabel="Mois", ylabel="Heure (UTC)", title="Irradiance moyenne (W/m²)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(FIGURES / "eda_heatmap.png", dpi=150); plt.close(fig)

    # 4. Irradiance vs nébulosité (heures de jour, 11h-13h)
    noon = df[df["hour"].between(11, 13)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(noon["cloud"], noon["irradiance"], s=3, alpha=.3)
    ax.set(xlabel="Nébulosité (%)", ylabel="Irradiance (W/m²)", title="Irradiance vs nébulosité (midi)")
    corr = noon[["cloud", "irradiance"]].corr().iloc[0, 1]
    ax.text(.05, .05, f"corr = {corr:.2f}", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(FIGURES / "eda_irradiance_vs_cloud.png", dpi=150); plt.close(fig)

    print("Statistiques utiles pour le rapport :")
    print(f"  Période : {df.index.min()} -> {df.index.max()} ({len(df)} heures)")
    print(f"  Part d'heures de nuit (irradiance≈0) : {(df['irradiance'] < 5).mean():.1%}")
    print(f"  Production moyenne : {df['pv_kwh'].mean():.4f} kWh/foyer/h | max : {df['pv_kwh'].max():.3f}")
    print(f"  Corrélation irradiance/nébulosité (midi) : {corr:.2f}")
    print(f"Figures -> {FIGURES}")


if __name__ == "__main__":
    main()
