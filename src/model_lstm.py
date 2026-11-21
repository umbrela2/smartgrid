"""
model_lstm.py - Prévision de la production photovoltaïque par LSTM (Partie B).

Usage :
    python src/model_lstm.py --sample      # test rapide (6 mois, 5 époques)
    python src/model_lstm.py               # entraînement complet

Ce que fait le script :
    1. charge data/processed/solar_pv.csv (produit par solar_loader.py)
    2. construit les features (irradiance, temp, nébulosité, heure/mois en sin/cos)
    3. découpe CHRONOLOGIQUEMENT en train/val/test (70/15/15)
    4. normalise (scaler ajusté sur le train uniquement)
    5. crée des fenêtres glissantes : 24 h passées -> 6 h futures
    6. évalue 2 baselines (persistance J-1, régression linéaire) puis entraîne le LSTM
    7. sauvegarde modèle, scalers, métriques, figures et data/processed/pred_pv.csv

Fonction exposée pour les autres membres du groupe :
    predict_pv(df, horizon=1) -> pd.Series  (index = timestamp prédit, valeurs en kWh/foyer/h)
"""
import argparse
import json
import random

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from solar_config import (BATCH_SIZE, DATA_PROC, DAY_THRESHOLD_WM2, EPOCHS, FEATURES, FIGURES,
                          HORIZON, LEARNING_RATE, LOOKBACK, MODELS, REPORTS, SEED, TRAIN_FRAC,
                          VAL_FRAC, ensure_dirs)

MODEL_PATH = MODELS / "lstm.keras"
SCALER_X_PATH = MODELS / "lstm_scaler_x.pkl"
META_PATH = MODELS / "lstm_meta.json"


# ----------------------------------------------------------------------------
# Données et features
# ----------------------------------------------------------------------------
def set_seeds():
    import tensorflow as tf
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute l'encodage cyclique de l'heure et du mois (le modèle comprend que 23h est proche de 0h)."""
    df = df.copy()
    h, m = df.index.hour, df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)
    return df


def load_data(sample: bool = False) -> pd.DataFrame:
    path = DATA_PROC / "solar_pv.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable. Lance d'abord : python src/solar_loader.py")
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    if sample:
        df = df.iloc[: 24 * 180]
    return add_features(df)


def make_windows(arr: np.ndarray, lookback: int, horizon: int, targets: np.ndarray | None = None):
    """
    arr     : (n, n_features)
    Fenêtre i : entrées = lignes [i, i+lookback-1] ; cibles = lignes [i+lookback, i+lookback+horizon-1]
    Retourne X (N, lookback, f) et, si targets est fourni, Y (N, horizon).
    """
    n = len(arr)
    n_win = n - lookback - horizon + 1 if targets is not None else n - lookback + 1
    X = np.stack([arr[i:i + lookback] for i in range(n_win)])
    if targets is None:
        return X
    Y = np.stack([targets[i + lookback:i + lookback + horizon] for i in range(n_win)])
    return X, Y


# ----------------------------------------------------------------------------
# Métriques
# ----------------------------------------------------------------------------
def metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, day_mask: np.ndarray) -> dict:
    """Métriques globales (24 h/24) et de jour uniquement (la nuit est triviale : production = 0)."""
    yt, yp, d = y_true.ravel(), y_pred.ravel(), day_mask.ravel()
    return {
        "model": name,
        "MAE": mean_absolute_error(yt, yp),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "R2": r2_score(yt, yp),
        "MAE_day": mean_absolute_error(yt[d], yp[d]),
        "RMSE_day": float(np.sqrt(mean_squared_error(yt[d], yp[d]))),
        "R2_day": r2_score(yt[d], yp[d]),
    }


# ----------------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------------
def build_model(n_features: int):
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([
        layers.Input(shape=(LOOKBACK, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(HORIZON),
    ])
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    return model


# ----------------------------------------------------------------------------
# Entraînement + évaluation
# ----------------------------------------------------------------------------
def train(sample: bool = False, epochs: int | None = None):
    from tensorflow import keras
    ensure_dirs()
    set_seeds()
    epochs = epochs or (5 if sample else EPOCHS)

    df = load_data(sample)
    n = len(df)
    i_train, i_val = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    print(f"Données : {n} heures | train < {df.index[i_train]} | val < {df.index[i_val]} | test ensuite")

    # --- normalisation : AJUSTÉE SUR LE TRAIN UNIQUEMENT (pas de fuite de données)
    scaler_x = MinMaxScaler().fit(df[FEATURES].iloc[:i_train])
    pv_max = float(df["pv_kwh"].iloc[:i_train].max())
    X_all = scaler_x.transform(df[FEATURES])
    y_all = df["pv_kwh"].values / pv_max            # cible mise à l'échelle [0, ~1]
    irr = df["irradiance"].values

    Xw, Yw = make_windows(X_all, LOOKBACK, HORIZON, y_all)
    _, Iw = make_windows(X_all, LOOKBACK, HORIZON, irr)       # irradiance aux heures cibles (masque jour)
    # Persistance J-1 : valeur de production 24 h avant l'heure cible
    _, Pw = make_windows(X_all, LOOKBACK, HORIZON, np.roll(y_all, 24))
    n_win = len(Xw)
    first = np.arange(n_win) + LOOKBACK            # index de la 1re heure cible
    last = first + HORIZON - 1                     # index de la dernière heure cible
    m_train = last < i_train
    m_val = (first >= i_train) & (last < i_val)
    m_test = first >= i_val
    print(f"Fenêtres : train={m_train.sum()} val={m_val.sum()} test={m_test.sum()} (fenêtres à cheval sur 2 périodes écartées)")

    # --- baselines
    def to_kwh(a):
        return np.clip(a, 0, None) * pv_max

    Yt_test, day_test = to_kwh(Yw[m_test]), Iw[m_test] > DAY_THRESHOLD_WM2
    results = []
    results.append(metrics("Persistance J-1", Yt_test, to_kwh(Pw[m_test]), day_test))
    lin = LinearRegression().fit(Xw[m_train].reshape(m_train.sum(), -1), Yw[m_train])
    lin_pred_test = lin.predict(Xw[m_test].reshape(m_test.sum(), -1))
    results.append(metrics("Régression linéaire", Yt_test, to_kwh(lin_pred_test), day_test))

    # --- LSTM
    model = build_model(len(FEATURES))
    model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss"),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_loss"),
    ]
    hist = model.fit(Xw[m_train], Yw[m_train], validation_data=(Xw[m_val], Yw[m_val]),
                     epochs=epochs, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)

    pred_all = model.predict(Xw, batch_size=512, verbose=0)
    pred_test = pred_all[m_test]
    results.append(metrics("LSTM", Yt_test, to_kwh(pred_test), day_test))

    table = pd.DataFrame(results).set_index("model")
    print("\n=== RÉSULTATS SUR LE TEST (kWh/foyer/h) ===")
    print(table.round(4).to_string())
    table.round(5).to_csv(REPORTS / "metrics_pv.csv")

    # MAE par horizon (t+1 ... t+6)
    mae_h = {name: [mean_absolute_error(Yt_test[:, k], to_kwh(p[:, k])) for k in range(HORIZON)]
             for name, p in [("Persistance J-1", Pw[m_test]), ("Régression linéaire", lin_pred_test), ("LSTM", pred_test)]}
    pd.DataFrame(mae_h, index=[f"t+{k+1}" for k in range(HORIZON)]).round(5).to_csv(REPORTS / "metrics_pv_by_horizon.csv")

    # --- sauvegardes
    joblib.dump(scaler_x, SCALER_X_PATH)
    META_PATH.write_text(json.dumps({"pv_max": pv_max, "lookback": LOOKBACK, "horizon": HORIZON,
                                     "features": FEATURES, "epochs_run": len(hist.history["loss"])}, indent=2))

    # --- fichier de prédictions pour les autres membres (horizon t+1)
    split = np.where(m_train, "train", np.where(m_val, "val", np.where(m_test, "test", "boundary")))
    pred_df = pd.DataFrame({
        "timestamp": df.index[first],
        "y_pred": to_kwh(pred_all[:, 0]),
        "y_true": to_kwh(Yw[:, 0]),
        "split": split,
    })
    pred_df = pred_df[pred_df["split"] != "boundary"]
    pred_df.to_csv(DATA_PROC / "pred_pv.csv", index=False)
    print(f"Prédictions -> {DATA_PROC / 'pred_pv.csv'} ({len(pred_df)} lignes)")

    make_figures(hist, df, first[m_test], Yt_test, to_kwh(pred_test), to_kwh(Pw[m_test]), to_kwh(lin_pred_test),
                 day_test, mae_h)
    print(f"Figures -> {FIGURES}")
    return table


# ----------------------------------------------------------------------------
# Figures pour le rapport
# ----------------------------------------------------------------------------
def make_figures(hist, df, first_test, Yt, P_lstm, P_pers, P_lin, day_mask, mae_h):
    # 1. courbes de loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist.history["loss"], label="train"); ax.plot(hist.history["val_loss"], label="validation")
    ax.set(xlabel="Époque", ylabel="MSE (échelle normalisée)", title="Courbes d'apprentissage du LSTM")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGURES / "lstm_loss.png", dpi=150); plt.close(fig)

    # 2. prédit vs réel (t+1) sur la semaine la plus ensoleillée et la moins productive du test
    ts = df.index[first_test]
    s = pd.DataFrame({"true": Yt[:, 0], "lstm": P_lstm[:, 0], "pers": P_pers[:, 0]}, index=ts)
    if len(s) > 168:
        weekly = s["true"].rolling(168).sum().dropna()
        for label, end in [("semaine la plus productive", weekly.idxmax()), ("semaine la moins productive", weekly.idxmin())]:
            w = s.loc[end - pd.Timedelta(hours=167): end]
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(w.index, w["true"], label="Réel", lw=2)
            ax.plot(w.index, w["lstm"], label="LSTM (t+1)", ls="--")
            ax.plot(w.index, w["pers"], label="Persistance J-1", ls=":", alpha=.7)
            ax.set(ylabel="kWh/foyer/h", title=f"Production PV : prédit vs réel ({label})")
            ax.legend(); ax.grid(alpha=.3); fig.autofmt_xdate(); fig.tight_layout()
            fig.savefig(FIGURES / f"lstm_semaine_{'max' if 'plus' in label else 'min'}.png", dpi=150); plt.close(fig)

    # 3. erreur moyenne par heure du jour (t+1)
    hours = ts.hour
    err = pd.DataFrame({"LSTM": np.abs(Yt[:, 0] - P_lstm[:, 0]), "Persistance J-1": np.abs(Yt[:, 0] - P_pers[:, 0]),
                        "hour": hours}).groupby("hour").mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    err.plot(ax=ax, marker="o", ms=3)
    ax.set(xlabel="Heure (UTC)", ylabel="MAE (kWh/foyer/h)", title="Erreur absolue moyenne par heure (t+1)")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGURES / "lstm_erreur_par_heure.png", dpi=150); plt.close(fig)

    # 4. erreur par horizon
    fig, ax = plt.subplots(figsize=(7, 4))
    pd.DataFrame(mae_h, index=[f"t+{k+1}" for k in range(HORIZON)]).plot.bar(ax=ax, rot=0)
    ax.set(ylabel="MAE (kWh/foyer/h)", title="Erreur selon l'horizon de prévision")
    fig.tight_layout(); fig.savefig(FIGURES / "lstm_erreur_par_horizon.png", dpi=150); plt.close(fig)

    # 5. nuage de points prédit vs réel
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Yt[:, 0][day_mask[:, 0]], P_lstm[:, 0][day_mask[:, 0]], s=4, alpha=.3)
    mx = max(Yt.max(), P_lstm.max())
    ax.plot([0, mx], [0, mx], "r--", lw=1)
    ax.set(xlabel="Réel (kWh/foyer/h)", ylabel="Prédit LSTM (t+1)", title="Prédit vs réel (heures de jour)")
    fig.tight_layout(); fig.savefig(FIGURES / "lstm_scatter.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# Interface pour les autres membres (contrat d'interface)
# ----------------------------------------------------------------------------
def predict_pv(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Prévoit la production PV (kWh par foyer et par heure).

    df      : DataFrame horaire (index datetime UTC, ou colonne 'timestamp') contenant au minimum
              les colonnes 'irradiance', 'temp', 'cloud'. Il faut au moins LOOKBACK (=24) lignes.
    horizon : 1..HORIZON. 1 = prévision pour l'heure qui suit la dernière heure de la fenêtre.
    Retourne une Series indexée par l'horodatage PRÉDIT (= heure de la fenêtre + horizon heures).
    """
    from tensorflow import keras
    if not 1 <= horizon <= HORIZON:
        raise ValueError(f"horizon doit être entre 1 et {HORIZON}")
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"]))
    df = add_features(df.sort_index())
    if len(df) < LOOKBACK:
        raise ValueError(f"Il faut au moins {LOOKBACK} lignes horaires consécutives.")
    meta = json.loads(META_PATH.read_text())
    scaler_x = joblib.load(SCALER_X_PATH)
    model = keras.models.load_model(MODEL_PATH)
    X = make_windows(scaler_x.transform(df[FEATURES]), LOOKBACK, HORIZON)   # (N, lookback, f)
    pred = np.clip(model.predict(X, batch_size=512, verbose=0)[:, horizon - 1], 0, None) * meta["pv_max"]
    idx = df.index[LOOKBACK - 1:] + pd.Timedelta(hours=horizon)
    return pd.Series(pred, index=idx, name="pv_pred")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="mode rapide : 6 mois de données, 5 époques")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    train(sample=args.sample, epochs=args.epochs)
