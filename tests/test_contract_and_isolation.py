"""Output-contract and isolation checks (no raw Kaggle data needed)."""
import ast
import re
from pathlib import Path

import pandas as pd

from src.part_a.export_predictions import COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CSV = ROOT / "data" / "processed" / "pred_consumption.csv"
ORIGINAL_COLUMNS = ["timestamp", "day", "predicted_kwh_per_household", "actual_kwh_per_household",
                    "error_kwh_per_household", "split", "n_households", "is_weekend", "is_holiday"]


def test_export_column_names_and_order_are_the_agent_c_contract():
    assert COLUMNS == ORIGINAL_COLUMNS


def test_shipped_example_csv_matches_contract():
    t = pd.read_csv(SHIPPED_CSV)
    assert list(t.columns) == ORIGINAL_COLUMNS
    assert set(t["split"]) == {"train", "test"} and not t.isna().any().any()
    assert t["day"].is_monotonic_increasing
    # Agent C rejects duplicated TEST days, so test days must be unique.
    assert t.loc[t.split == "test", "day"].is_unique
    # KNOWN, preserved behaviour of the original example file: two UK clock-change dates appear twice
    # (both in the train split). See README "Known limitations".
    assert sorted(t.loc[t["day"].duplicated(), "day"]) == ["2012-03-25", "2013-03-31"]
    # train rows strictly precede test rows (chronological split)
    assert t.loc[t.split == "train", "day"].max() < t.loc[t.split == "test", "day"].min()
    assert t["is_weekend"].isin([0, 1]).all() and t["is_holiday"].isin([0, 1]).all()
    assert (t["predicted_kwh_per_household"] >= 0).all()
    assert ((t["predicted_kwh_per_household"] - t["actual_kwh_per_household"]) - t["error_kwh_per_household"]).abs().max() < 1e-9


def test_shipped_solar_csv_has_the_columns_agent_a_reads():
    cols = pd.read_csv(ROOT / "data" / "processed" / "solar_pv.csv", nrows=1).columns
    assert {"timestamp", "irradiance", "temp", "cloud", "pv_kwh"} <= set(cols)


def _py_files():
    return [p for p in (ROOT / "src").rglob("*.py")] + [p for p in (ROOT / "tests").glob("*.py")]


def test_no_import_of_other_agents_or_integrated_project():
    """AST check of real import statements (docstrings may mention Part B/C in prose)."""
    bad = re.compile(r"(part_b|part_c|pv_forecast|pricing_agent|community|simulation|streamlit|ollama|llm)", re.I)
    offenders = []
    for p in _py_files():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            offenders += [(p.name, n) for n in names if bad.search(n)]
    assert offenders == []


def test_no_hardcoded_absolute_or_original_project_paths():
    pat = re.compile(r"(smartgrid_final|agent_a_handoff|[A-Za-z]:\\\\|/home/|/mnt/|/Users/)")
    hits = [p.name for p in (ROOT / "src").rglob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert hits == []


def test_no_agent_b_or_c_source_files_present():
    names = {p.name for p in (ROOT / "src").rglob("*.py")}
    assert not names & {"pv_forecast.py", "pricing_agent.py", "simulation.py", "community.py",
                        "load_profile.py", "data_interface.py", "scenario.py", "commands.py"}
    assert not (ROOT / "src" / "part_c").exists() and not (ROOT / "src" / "llm").exists()
    assert not (ROOT / "models" / "pricing_agent.joblib").exists()
