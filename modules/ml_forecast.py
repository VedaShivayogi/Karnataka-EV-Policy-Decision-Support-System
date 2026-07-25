"""
ML Forecast
Predicts district-level EV adoption rate using XGBoost/LightGBM/Scikit-learn,
based on features like fuel price, subsidy amount, charging stations, income level.

Run:
    python modules/ml_forecast.py
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


FEATURES = ["fuel_price_inr", "subsidy_inr", "charging_stations", "avg_income_inr", "urbanization_pct"]
TARGET = "ev_adoption_pct"


def load_data(path=os.path.join(config.DATA_DIR, "district_data.csv")):
    return pd.read_csv(path)


def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if HAS_XGB:
        model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Model: {'XGBoost' if HAS_XGB else 'RandomForest (fallback)'} | MAE={mae:.2f} | R2={r2:.2f}")
    return model


def predict_scenario(model, base_row: dict, changes: dict = None):
    """Predict EV adoption % for a district, optionally applying what-if changes."""
    row = base_row.copy()
    if changes:
        row.update(changes)
    X = pd.DataFrame([{k: row[k] for k in FEATURES}])
    pred = model.predict(X)[0]
    return float(pred)


if __name__ == "__main__":
    df = load_data()
    model = train_model(df)

    sample = df.iloc[0].to_dict()
    baseline = predict_scenario(model, sample)
    boosted = predict_scenario(model, sample, {"subsidy_inr": sample["subsidy_inr"] + 5000})
    print(f"\nDistrict: {sample.get('district', 'sample')}")
    print(f"Baseline predicted EV adoption: {baseline:.2f}%")
    print(f"With +₹5,000 subsidy:           {boosted:.2f}%  (Δ {boosted - baseline:+.2f} pts)")
