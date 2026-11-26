# Agent A — Electricity Consumption Forecasting

Standalone repository for **Agent A** of the university Smart Grid project. It trains and runs a
Ridge-regression model that predicts the **mean daily electricity consumption per household (kWh)**
and exports the result to `data/processed/pred_consumption.csv`.

## 1. Role in the Smart Grid project

The project has three independent agents that only exchange CSV files:

| Agent | Responsibility | Output |
|-------|---------------|--------|
| **A (this repo)** | Daily electricity consumption forecast | `pred_consumption.csv` |
| B | Hourly solar PV forecast (owned by another teammate) | `solar_pv.csv`, `pred_pv.csv` |
| C | Dynamic pricing, community simulation, evaluation | reads A's and B's CSVs |

### Independence
- This repo contains **no code or model from Agent B or Agent C** and imports nothing from them
  (enforced by `tests/test_contract_and_isolation.py`).
- Agent A **does need one CSV produced by Agent B: `data/processed/solar_pv.csv`**. It supplies the
  daily temperature (max/min) and cloud cover features. It is needed for preprocessing, training and
  exporting predictions. A copy is included here so the repo works out of the box; if Agent B's owner
  regenerates it, replace the file.
- Agent C only reads `data/processed/pred_consumption.csv`. It never calls Agent A's code.

## 2. Folder structure

```
Agent_A_GitHub/
├── README.md
├── requirements.txt            runtime dependencies (scikit-learn pinned to the model's version)
├── requirements-dev.txt        runtime + pytest
├── pytest.ini                  makes `pytest` work from the repo root
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── utils.py                path constants (ROOT, RAW, MODELS), SEED, regression_metrics
│   └── part_a/
│       ├── data_loader.py      load raw Kaggle files + Agent B's solar_pv.csv
│       ├── preprocessing.py    clean, aggregate, merge, feature engineering, chronological split
│       ├── model_regression.py train / save / load / predict (StandardScaler + Ridge)
│       └── export_predictions.py  CLI that writes pred_consumption.csv
├── tests/                      24 self-contained tests (synthetic data, no Kaggle files needed)
├── models/consumption_regression.joblib   trained model (kept in git)
└── data/
    ├── raw/                    put the 3 Kaggle files here (git-ignored, see data/raw/README.md)
    └── processed/
        ├── solar_pv.csv        INPUT from Agent B (hourly, UTC)
        └── pred_consumption.csv  OUTPUT of Agent A (example, produced by the original project)
```

## 3. Required input files

| File | Location | Source | Included? |
|------|----------|--------|-----------|
| `daily_dataset.csv` | `data/raw/` | Kaggle *Smart meters in London* (`jeanmidev/smart-meters-in-london`) | **No** (large, third-party) |
| `weather_daily_darksky.csv` | `data/raw/` | same Kaggle dataset | **No** |
| `uk_bank_holidays.csv` | `data/raw/` | same Kaggle dataset | **No** |
| `solar_pv.csv` | `data/processed/` | **Agent B** (columns `timestamp, irradiance, temp, cloud, pv_kwh`) | Yes (example copy, ~0.8 MB) |

Download from https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london (free Kaggle account
needed) and copy the three files into `data/raw/`. Check Kaggle's licence terms before redistributing them.
The raw files were **not** in the ZIP this repo was extracted from, so regenerating the output from real
data was not verified (only on synthetic data).

How the inputs are used:
- `daily_dataset.csv`: `LCLid`, `day`, `energy_sum`, `energy_count` → target.
- `weather_daily_darksky.csv`: `time`, `humidity`, `windSpeed`.
- `uk_bank_holidays.csv`: two columns (date, type) → `is_holiday`.
- `solar_pv.csv`: `temp` → `temperatureMax/Min`, `cloud` → `cloudCover`. (`irradiance` and `pv_kwh` must
  exist in the file but are not model features.)

## 4. Installation (Windows PowerShell)

Python 3.12 is the target (the code was tested on 3.13.18; see limitations).

```powershell
cd Agent_A_GitHub
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
Run every command below **from the repository root** (the folder containing `src/`).

## 5. Train the model

Needs the 3 raw files and `solar_pv.csv`.

```powershell
python -m src.part_a.model_regression
```

This builds the dataset, trains `StandardScaler + Ridge(alpha=1.0)` on the first 80 % of days
(chronological, never shuffled), prints the model metrics and the "same as last week" baseline
(`lag_7`) metrics, and **overwrites** `models/consumption_regression.joblib`.
Retraining is optional: the included model works as is.

## 6. Load the trained model

```python
import pandas as pd
from src.part_a.model_regression import predict_baseline_consumption
# features_df must contain the 13 FEATURES columns (see src/part_a/preprocessing.py)
kwh = predict_baseline_consumption(features_df)   # numpy array, kWh per household
```

Features: `temperatureMax, temperatureMin, humidity, windSpeed, cloudCover, dow_sin, dow_cos,
month_sin, month_cos, is_weekend, is_holiday, lag_1, lag_7`. `lag_1`/`lag_7` are the *measured*
consumption one day / one week earlier, so each prediction is a **one-day-ahead** forecast made with real
history, not a recursive multi-day forecast.

## 7. Export predictions

```powershell
python -m src.part_a.export_predictions                      # all days  -> data/processed/pred_consumption.csv
python -m src.part_a.export_predictions --split test         # held-out days only
python -m src.part_a.export_predictions --out data\processed\my.csv --model models\consumption_regression.joblib
python -m src.part_a.export_predictions --help
```

## 8. Output CSV schema (`data/processed/pred_consumption.csv`)

One row = one day (mean of one household). Column names and order are the contract with Agent C and
must not change.

| Column | Meaning |
|--------|---------|
| `timestamp` | start of day, `YYYY-MM-DD 00:00:00` (same text format as Agent B's CSV) |
| `day` | `YYYY-MM-DD` — **join key used by Agent C** |
| `predicted_kwh_per_household` | model output, kWh/household/day — **used by Agent C** |
| `actual_kwh_per_household` | measured value from the smart-meter data — used by Agent C for pricing-agent training and evaluation |
| `error_kwh_per_household` | `predicted − actual` (evaluation only) |
| `split` | `train` or `test` (same chronological 80/20 split used for training) — **used by Agent C** |
| `n_households` | number of meters averaged that day (information only) |
| `is_weekend` | 0/1 — **used by Agent C** to choose the weekday/weekend load shape |
| `is_holiday` | 0/1 (information only; Agent C parses it if present) |

How Agent C uses the file (read from the original project's `src/part_c/data_interface.py`):
- Required columns: `day`, `predicted_kwh_per_household`, `is_weekend`, `split`. It rejects negative
  predictions, `is_weekend` not in {0,1}, `split` values other than `train`/`test`, and duplicated test days.
- **Test rows** are the rows with `split == "test"`; final evaluation uses only those.
- **Matching with Agent B:** Agent B's hourly `pred_pv.csv` timestamps are truncated to the day and pivoted to a
  day × 24-hour table; only days with all 24 hours of the test split are kept. Agent C keeps the
  days that are in both Agent A's test rows (matched on `day`) and Agent B's complete test days.
- Agent C's pricing-agent training additionally reads `actual_kwh_per_household` (train and test rows).

## 9. Run the tests

```powershell
python -m pytest -q
```

All 24 tests use small synthetic data; no Kaggle files are needed. They cover cleaning, aggregation,
feature engineering, the chronological split, training vs the lag-7 baseline, the export table/CSV, the
shipped model's compatibility, the CSV contract, and the absence of Agent B/C imports and hard-coded paths.

## 10. Upload to GitHub (PowerShell)

Create an empty repository on GitHub first, then, inside `Agent_A_GitHub/`:

```powershell
git init
git add .
git commit -m "Initial commit: Agent A consumption forecasting"
git branch -M main
git remote add origin <THEIR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Replace `<THEIR_GITHUB_REPOSITORY_URL>` with your own repository URL.

## 11. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run from the repo root and use `python -m ...` (not `python src\part_a\...py`). |
| `Missing input file: ...daily_dataset.csv` | Put the 3 Kaggle files in `data/raw/` (section 3). |
| `Missing input file: ...solar_pv.csv` | Get Agent B's `solar_pv.csv` and put it in `data/processed/`. |
| `FileNotFoundError: Part A model not found` | Train first (section 5) or restore `models/consumption_regression.joblib`. |
| `InconsistentVersionWarning` when loading the model | Install the pinned version: `pip install scikit-learn==1.9.1`, or retrain. |
| `KeyError: 'pv_kwh'` / `'irradiance'` | `solar_pv.csv` must have columns `timestamp, irradiance, temp, cloud, pv_kwh`. |
| PowerShell refuses to activate `.venv` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## 12. Known limitations
- Two dates (2012-03-25, 2013-03-31; UK clock changes) appear twice in the example `pred_consumption.csv`
  (both in the train split). This comes from the original code path (most likely duplicated dates in the
  weather file creating duplicate rows in the merge; not verifiable without the raw data) and was **preserved**
  because fixing it would change the split and the outputs. Agent C only rejects duplicated *test* days, so it
  is unaffected.
- Python 3.12 and pandas 2.x were not available for testing; tests ran on Python 3.13.18, pandas 3.0.5, numpy 2.5.3,
  scikit-learn 1.9.1.
- The solar file's data source/licence is Agent B's responsibility.
