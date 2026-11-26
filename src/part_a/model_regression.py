"""Step 5-6 - TRAIN, EVALUATE, SAVE, and PREDICT (the function C will call)."""
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .preprocessing import FEATURES, TARGET, time_split
from ..utils import MODELS, regression_metrics

def train(df: pd.DataFrame):
    train_df, test_df = time_split(df)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))  # linear regression + light regularisation
    model.fit(train_df[FEATURES], train_df[TARGET])
    # Baseline to beat: "tomorrow = same as last week"
    baseline = regression_metrics(test_df[TARGET], test_df["lag_7"])
    metrics = regression_metrics(test_df[TARGET], model.predict(test_df[FEATURES]))
    return model, metrics, baseline, test_df

def save(model, path=MODELS / "consumption_regression.joblib"):
    MODELS.mkdir(exist_ok=True); joblib.dump(model, path)

def predict_baseline_consumption(features_df: pd.DataFrame, path=MODELS / "consumption_regression.joblib"):
    """INTERFACE FOR C: DataFrame with FEATURES columns -> kWh per household."""
    return joblib.load(path).predict(features_df[FEATURES])

def main():
    from .data_loader import load_daily, load_weather, load_holidays, load_solar
    from .preprocessing import build_dataset
    data = build_dataset(load_daily(), load_weather(), load_holidays(), load_solar())
    model, m, b, _ = train(data)
    print("Regression:", m); print("Baseline lag_7:", b); save(model)

if __name__ == "__main__":
    main()
