import json
import requests
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score

RAW_URL = "https://raw.githubusercontent.com/BallesJr/polymarket-weather-edge/main/data/paper_portfolio_weather.json"

def load_trades() -> pd.DataFrame:
    resp = requests.get(RAW_URL)
    portfolio = resp.json()

    trades = portfolio["closed_trades"]
    df = pd.DataFrame(trades)

    # Only resolved trades (WON or LOST)
    df = df[df["status"].isin(["WON", "LOST"])].copy()

    # Target: 1 if WON, 0 if LOST
    df["won"] = (df["status"] == "WON").astype(int)

    return df

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = df.copy()

    # Encode direction as binary
    features["is_buy_yes"] = (features["direction"] == "BUY_YES").astype(int)

    # Enocode city as numeric categories
    features["city_code"] = features["series_slug"].astype("category").cat.codes

    # Encode data source
    features["is_observation"] = (features["data_source"] == "observation").astype(int)

    # Temperature difference(forecast vs observed, NaN if forecast)
    features["temp_diff"] = features["observed_max_c"] - features["forecast_temp_c"]

    # Use model_prob_gaussian if available, otherwise fall back to model_prob
    if "model_prob_gaussian" in features.columns:
        features["model_prob_gaussian"] = features["model_prob_gaussian"].fillna(features["model_prob"])
    else:
        features["model_prob_gaussian"] = features["model_prob"]

    feature_cols = ["entry_prob", "model_prob_gaussian", "forecast_horizon_days", "is_buy_yes", "city_code",
                "is_observation", "forecast_temp_c", "temp_diff"]
    
    X = features[feature_cols].copy()
    y = features["won"]

    # Fill NaNs with median (RF can't handle NaN)
    X = X.fillna(X.median())

    return X, y

def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5, # Limit depth to avoid overfitting with around 350 trades
        min_samples_leaf=10, # At least 10 samples per leaf
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Brier score: {brier:.4f} (lower is better, random = 0.25)")
    print(f"AUC-ROC:  {auc:.4f} (higher is better, random = 0.50)")

    # Feature importance
    for feat, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[-1]):
        print(f"{feat:<25} {imp:.3f}")

    os.makedirs("models", exist_ok=True)
    with open ("models/rf_weather.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to models/rf_weather.pkl")

    return model

if __name__ == "__main__":
    df = load_trades()
    X, y = preprocess(df)
    print(f"Features shape: {X.shape}")
    model = train_model(X, y)