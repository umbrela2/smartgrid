"""Typical 24h shape of household consumption (sums to 1), weekday vs weekend.
Best: measured from the Kaggle hhblock files. Fallback: a typical double-peak shape."""
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import ROOT

PROFILE_CSV = ROOT / "data" / "processed" / "load_profile.csv"

def _fallback(weekend: bool) -> np.ndarray:
    h = np.arange(24)
    g = lambda mu, sd: np.exp(-0.5 * ((h - mu) / sd) ** 2)
    base = 0.5 + 1.0 * g(18.5, 2.2) + (0.5 * g(11, 3.0) if weekend else 0.4 * g(7.5, 1.3))
    return base / base.sum()

def build_profile_from_blocks(block_dir=ROOT / "data" / "raw" / "hhblock_dataset",
                              max_blocks=10, out=PROFILE_CSV) -> pd.DataFrame:
    """Reads block_*.csv (LCLid, day, hh_0..hh_47), averages normalised daily curves."""
    cols = [f"hh_{i}" for i in range(48)]
    shapes = {True: [], False: []}
    for f in sorted(Path(block_dir).glob("block_*.csv"))[:max_blocks]:
        df = pd.read_csv(f, parse_dates=["day"]).dropna(subset=cols)
        hourly = df[cols].to_numpy().reshape(len(df), 24, 2).sum(axis=2)
        hourly = hourly / hourly.sum(axis=1, keepdims=True)
        wk = (df["day"].dt.dayofweek >= 5).to_numpy()
        for flag in (True, False):
            if (wk == flag).any(): shapes[flag].append(hourly[wk == flag])
    res = pd.DataFrame({"hour": range(24),
                        "weekday": np.vstack(shapes[False]).mean(0),
                        "weekend": np.vstack(shapes[True]).mean(0)})
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True); res.to_csv(out, index=False)
    return res

def get_profile(path=PROFILE_CSV):
    """Returns (profile dict, 'measured' | 'fallback')."""
    if Path(path).exists():
        p = pd.read_csv(path)
        return {"weekday": p["weekday"].to_numpy(), "weekend": p["weekend"].to_numpy()}, "measured"
    return {"weekday": _fallback(False), "weekend": _fallback(True)}, "fallback"

if __name__ == "__main__":
    print(build_profile_from_blocks().round(4))
