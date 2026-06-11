import requests
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime, timezone

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

from signal_engine import REGIME_START

RAW_URL = "https://raw.githubusercontent.com/BallesJr/polymarket-weather-edge/main/data/paper_portfolio_weather.json"

MIN_SPECIALIZED_SAMPLES = 150  # minimum BUY_NO T+0 trades to train and deploy a model

# Features for the general model (all trades)
ALL_FEATURES = ["entry_prob", "model_prob_gaussian", "forecast_horizon_days", "is_buy_yes",
                "city_code", "is_observation", "forecast_temp_c", "temp_diff"]

# Features for the specialized BUY_NO T+0 model (drops always-zero columns)
SPECIALIZED_FEATURES = ["entry_prob", "model_prob_gaussian", "city_code",
                        "is_observation", "forecast_temp_c", "temp_diff"]


def load_trades() -> pd.DataFrame:
    resp = requests.get(RAW_URL)
    portfolio = resp.json()
    df = pd.DataFrame(portfolio["closed_trades"])
    df = df[df["status"].isin(["WON", "LOST"])].copy()
    # Clean-data regime only: trades before REGIME_START ran under known data
    # bugs (wrong Paris station, NaN gaussian on Fahrenheit markets, model
    # observations instead of station METARs) and would teach the model noise
    df = df[df["opened_at"] >= REGIME_START].copy()
    df["won"] = (df["status"] == "WON").astype(int)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    features = df.copy()
    features["is_buy_yes"] = (features["direction"] == "BUY_YES").astype(int)
    # Fixed category list so the slug->code mapping can be persisted with the
    # model and reproduced exactly at inference time
    city_categories = sorted(features["series_slug"].unique())
    features["city_code"] = pd.Categorical(features["series_slug"], categories=city_categories).codes
    features["is_observation"] = (features["data_source"] == "observation").astype(int)
    features["temp_diff"] = features["observed_max_c"] - features["forecast_temp_c"]

    if "model_prob_gaussian" in features.columns:
        features["model_prob_gaussian"] = features["model_prob_gaussian"].fillna(features["model_prob"])
    else:
        features["model_prob_gaussian"] = features["model_prob"]

    X = features[ALL_FEATURES].copy().fillna(features[ALL_FEATURES].median())
    y = features["won"]
    return X, y, city_categories


def calibration_diagnostic(probs: np.ndarray, y: pd.Series, label: str = "") -> None:
    if label:
        print(f"\n[{label}] Calibration diagnostic (predicted prob vs actual win rate)")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    labels = ["<0.10","0.10-0.20","0.20-0.30","0.30-0.40","0.40-0.50","0.50-0.60","0.60-0.70",">0.70"]
    df_cal = pd.DataFrame({"prob": probs, "won": y.values})
    df_cal["bucket"] = pd.cut(df_cal["prob"], bins=bins, labels=labels)
    g = df_cal.groupby("bucket", observed=True).agg(
        n=("won", "count"),
        avg_pred=("prob", "mean"),
        win_rate=("won", "mean"),
    )
    g["deviation"] = g["avg_pred"] - g["win_rate"]
    print(g.to_string())


def _cv_brier(model, X, y, cv=5) -> tuple[float, float]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_brier_score")
    return -scores.mean(), scores.std()


def _best_model(X: pd.DataFrame, y: pd.Series, feat_cols: list, label: str) -> tuple:
    rf_params = dict(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42)
    brier_raw, _ = _cv_brier(RandomForestClassifier(**rf_params), X, y)
    brier_sig, _ = _cv_brier(CalibratedClassifierCV(RandomForestClassifier(**rf_params), method="sigmoid",  cv=5), X, y)
    brier_iso, _ = _cv_brier(CalibratedClassifierCV(RandomForestClassifier(**rf_params), method="isotonic", cv=5), X, y)
    auc = cross_val_score(RandomForestClassifier(**rf_params), X, y, cv=5, scoring="roc_auc").mean()
    print(f"  [{label}]  raw={brier_raw:.4f}  sig={brier_sig:.4f}  iso={brier_iso:.4f}  AUC={auc:.4f}")

    best_brier = min(brier_raw, brier_sig, brier_iso)
    best_method = {brier_raw: "raw", brier_sig: "sigmoid", brier_iso: "isotonic"}[best_brier]

    if best_method == "raw":
        model = RandomForestClassifier(**rf_params)
    else:
        model = CalibratedClassifierCV(RandomForestClassifier(**rf_params), method=best_method, cv=5)
    model.fit(X, y)
    return model, best_method, best_brier, feat_cols


def train_model(df_all: pd.DataFrame, X_all: pd.DataFrame, y_all: pd.Series, city_categories: list):
    print("=" * 60)
    print("CV Brier comparison (5-fold, lower = better)")
    print("=" * 60)

    # Specialized model: BUY_NO T+0 only (under the current regime this is the
    # entire population — every trade the bot opens is BUY_NO T+0)
    mask = (df_all["direction"] == "BUY_NO") & (df_all["forecast_horizon_days"] == 0)
    n_spec = int(mask.sum())
    print(f"  BUY_NO T+0 samples: {n_spec} (min required: {MIN_SPECIALIZED_SAMPLES})")

    if n_spec < MIN_SPECIALIZED_SAMPLES:
        print("\nInsufficient clean data — model NOT saved, existing file left untouched.")
        return None, None

    X_spec = X_all[mask][SPECIALIZED_FEATURES]
    y_spec = y_all[mask]
    model, method, brier, feats = _best_model(
        X_spec, y_spec, SPECIALIZED_FEATURES, "BUY_NO T+0 specialized"
    )
    print(f"\nWinner: specialized BUY_NO T+0 model ({method}, Brier {brier:.4f})")

    # Feature importances
    base_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42)
    base_rf.fit(X_spec, y_spec)
    print("\nFeature importances:")
    for feat, imp in sorted(zip(X_spec.columns, base_rf.feature_importances_), key=lambda x: -x[-1]):
        print(f"  {feat:<25} {imp:.3f}")

    calibration_diagnostic(base_rf.predict_proba(X_spec)[:, 1], y_spec, label="final model (in-sample)")

    # Self-describing payload: everything inference needs to rebuild the exact
    # feature row, including the slug->city_code mapping used in training
    payload = {
        "model": model,
        "features": feats,
        "city_categories": city_categories,
        "n_train": n_spec,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "regime_start": REGIME_START,
    }
    os.makedirs("models", exist_ok=True)
    with open("models/rf_weather.pkl", "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved: models/rf_weather.pkl  [features: {feats}, n_train: {n_spec}]")

    return model, feats


if __name__ == "__main__":
    df = load_trades()
    n_spec = len(df[(df["direction"] == "BUY_NO") & (df["forecast_horizon_days"] == 0)])
    print(f"Clean-regime resolved trades (opened >= {REGIME_START}): {len(df)}  [BUY_NO T+0: {n_spec}]")

    if n_spec < MIN_SPECIALIZED_SAMPLES:
        print(f"Need {MIN_SPECIALIZED_SAMPLES} clean BUY_NO T+0 trades to train; "
              f"have {n_spec}. Nothing trained, existing model file untouched.")
        sys.exit(0)

    X, y, city_categories = preprocess(df)
    train_model(df, X, y, city_categories)
