# Smart Grid - Part A (data + regression)
1. Download Kaggle "smart-meters-in-london" -> put daily_dataset.csv, weather_daily_darksky.csv, uk_bank_holidays.csv in data/raw/
2. pip install pandas scikit-learn joblib pytest matplotlib
3. python -m src.model_regression   (trains + saves models/consumption_regression.joblib)
4. pytest
C uses: from src.model_regression import predict_baseline_consumption
