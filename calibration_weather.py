import requests
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

RAW_URL = "https://raw.githubusercontent.com/BallesJr/polymarket-weather-edge/main/data/paper_portfolio_weather.json"

MIN_SPECIALIZED_SAMPLES = 150  # minimum BUY_NO T+0 trades to use the specialized model

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
    df["won"] = (df["status"] == "WON").astype(int)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["is_buy_yes"]     = (features["direction"] == "BUY_YES").astype(int)
    features["city_code"]      = features["series_slug"].astype("category").cat.codes
    features["is_observation"] = (features["data_source"] == "observation").astype(int)
    features["temp_diff"]      = features["observed_max_c"] - features["forecast_temp_c"]

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


def train_model(df_all: pd.DataFrame, X_all: pd.DataFrame, y_all: pd.Series):
    print("=" * 60)
    print("CV Brier comparison (5-fold, lower = better)")
    print("=" * 60)

    # General model (all trades)
    model_gen, method_gen, brier_gen, feats_gen = _best_model(
        X_all[ALL_FEATURES], y_all, ALL_FEATURES, "all trades"
    )

    # Specialized model: BUY_NO T+0 only
    mask = (df_all["direction"] == "BUY_NO") & (df_all["forecast_horizon_days"] == 0)
    df_spec = df_all[mask].copy()
    n_spec = len(df_spec)
    print(f"\n  BUY_NO T+0 samples: {n_spec} (min required: {MIN_SPECIALIZED_SAMPLES})")

    if n_spec >= MIN_SPECIALIZED_SAMPLES:
        X_spec_raw = X_all[mask][SPECIALIZED_FEATURES]
        y_spec     = y_all[mask]
        model_spec, method_spec, brier_spec, feats_spec = _best_model(
            X_spec_raw, y_spec, SPECIALIZED_FEATURES, "BUY_NO T+0 specialized"
        )
        print(f"\nWinner: specialized BUY_NO T+0 model ({method_spec}, Brier {brier_spec:.4f})")
        final_model, final_feats = model_spec, feats_spec
        X_best, y_best = X_spec_raw, y_spec
    else:
        print(f"\nInsufficient data for specialized model — using general model ({method_gen}, Brier {brier_gen:.4f})")
        final_model, final_feats = model_gen, feats_gen
        X_best, y_best = X_all[ALL_FEATURES], y_all

    # Feature importances
    base_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42)
    base_rf.fit(X_best, y_best)
    print("\nFeature importances:")
    for feat, imp in sorted(zip(X_best.columns, base_rf.feature_importances_), key=lambda x: -x[-1]):
        print(f"  {feat:<25} {imp:.3f}")

    calibration_diagnostic(base_rf.predict_proba(X_best)[:, 1], y_best, label="final model (in-sample)")

    os.makedirs("models", exist_ok=True)
    with open("models/rf_weather.pkl", "wb") as f:
        pickle.dump((final_model, final_feats), f)
    print(f"\nSaved: models/rf_weather.pkl  [features: {final_feats}]")

    return final_model, final_feats


if __name__ == "__main__":
    df = load_trades()
    X, y = preprocess(df)
    print(f"Total resolved trades: {len(df)}")
    train_model(df, X, y)
