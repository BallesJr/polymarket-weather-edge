import requests
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

RAW_URL = "https://raw.githubusercontent.com/BallesJr/polymarket-weather-edge/main/data/paper_portfolio_weather.json"

ALL_FEATURES     = ["entry_prob", "model_prob_gaussian", "forecast_horizon_days", "is_buy_yes",
                    "city_code", "is_observation", "forecast_temp_c", "temp_diff"]
NO_ENTRY_FEATURES = [f for f in ALL_FEATURES if f != "entry_prob"]

def load_trades() -> pd.DataFrame:
    resp = requests.get(RAW_URL)
    portfolio = resp.json()
    df = pd.DataFrame(portfolio["closed_trades"])
    df = df[df["status"].isin(["WON", "LOST"])].copy()
    df["won"] = (df["status"] == "WON").astype(int)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["is_buy_yes"]   = (features["direction"] == "BUY_YES").astype(int)
    features["city_code"]    = features["series_slug"].astype("category").cat.codes
    features["is_observation"] = (features["data_source"] == "observation").astype(int)
    features["temp_diff"]    = features["observed_max_c"] - features["forecast_temp_c"]

    if "model_prob_gaussian" in features.columns:
        features["model_prob_gaussian"] = features["model_prob_gaussian"].fillna(features["model_prob"])
    else:
        features["model_prob_gaussian"] = features["model_prob"]

    X = features[ALL_FEATURES].copy().fillna(features[ALL_FEATURES].median())
    y = features["won"]
    return X, y

def calibration_diagnostic(probs: np.ndarray, y: pd.Series, label: str = "") -> None:
    if label:
        print(f"\n[{label}] Calibration diagnostic (predicted prob vs actual win rate)")
    bins   = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
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

def train_model(X: pd.DataFrame, y: pd.Series):
    rf_params = dict(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42)

    X_all    = X[ALL_FEATURES]
    X_no_ep  = X[NO_ENTRY_FEATURES]

    print("=" * 60)
    print("CV Brier comparison (5-fold, lower = better)")
    print("=" * 60)

    results = {}
    for label, Xv in [("with entry_prob", X_all), ("no entry_prob", X_no_ep)]:
        brier_raw, std_raw = _cv_brier(RandomForestClassifier(**rf_params), Xv, y)
        brier_sig, std_sig = _cv_brier(CalibratedClassifierCV(RandomForestClassifier(**rf_params), method="sigmoid",  cv=5), Xv, y)
        brier_iso, std_iso = _cv_brier(CalibratedClassifierCV(RandomForestClassifier(**rf_params), method="isotonic", cv=5), Xv, y)
        auc, _ = (-x for x in cross_val_score(RandomForestClassifier(**rf_params), Xv, y, cv=5, scoring="neg_brier_score")), None
        auc_mean = cross_val_score(RandomForestClassifier(**rf_params), Xv, y, cv=5, scoring="roc_auc").mean()
        print(f"\n  [{label}]")
        print(f"    raw RF:   Brier {brier_raw:.4f} ± {std_raw:.4f}  |  AUC {auc_mean:.4f}")
        print(f"    sigmoid:  Brier {brier_sig:.4f} ± {std_sig:.4f}")
        print(f"    isotonic: Brier {brier_iso:.4f} ± {std_iso:.4f}")
        results[label] = {"raw": brier_raw, "sigmoid": brier_sig, "isotonic": brier_iso, "auc": auc_mean}

    # Pick best combination: feature set + calibration method
    best_brier  = float("inf")
    best_config = ("with entry_prob", "raw", X_all, ALL_FEATURES)
    for feat_label, Xv, feat_cols in [("with entry_prob", X_all, ALL_FEATURES), ("no entry_prob", X_no_ep, NO_ENTRY_FEATURES)]:
        for method in ["raw", "sigmoid", "isotonic"]:
            b = results[feat_label][method]
            if b < best_brier:
                best_brier  = b
                best_config = (feat_label, method, Xv, feat_cols)

    best_feat_label, best_method, X_best, best_feat_cols = best_config
    print(f"\nWinner: [{best_feat_label}] + {best_method}  (Brier {best_brier:.4f})")

    # Feature importances of the winner's base RF
    base_rf = RandomForestClassifier(**rf_params)
    base_rf.fit(X_best, y)
    print(f"\nFeature importances ({best_feat_label}):")
    for feat, imp in sorted(zip(X_best.columns, base_rf.feature_importances_), key=lambda x: -x[-1]):
        print(f"  {feat:<25} {imp:.3f}")

    # Fit final production model
    if best_method == "raw":
        final_model = RandomForestClassifier(**rf_params)
    else:
        final_model = CalibratedClassifierCV(RandomForestClassifier(**rf_params), method=best_method, cv=5)
    final_model.fit(X_best, y)

    # Calibration diagnostics (in-sample)
    calibration_diagnostic(base_rf.predict_proba(X_best)[:, 1], y, label=f"raw RF ({best_feat_label})")
    calibration_diagnostic(final_model.predict_proba(X_best)[:, 1], y, label=f"final model ({best_method})")

    # Save model + feature list together to keep weather_api.py in sync
    os.makedirs("models", exist_ok=True)
    with open("models/rf_weather.pkl", "wb") as f:
        pickle.dump((final_model, best_feat_cols), f)
    print(f"\nSaved: models/rf_weather.pkl  [{best_method} | features: {best_feat_cols}]")

    return final_model, best_feat_cols

if __name__ == "__main__":
    df = load_trades()
    X, y = preprocess(df)
    print(f"Trades: {len(df)}  |  Features: {X.shape[1]}")
    train_model(X, y)
