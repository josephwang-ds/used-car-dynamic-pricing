#!/usr/bin/env python3
"""使用项目 .conda 环境训练 XGBoost 基线（供 Notebook 子进程或终端调用）。"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_fe.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
PRED_PATH = OUTPUT_DIR / "xgb_validation_predictions.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100
BIN_CAT_COLS = ["power_bin", "car_age_bin"]


def fit_bin_encoder(series: pd.Series) -> dict:
    cats = sorted(series.dropna().astype(str).unique())
    return {c: i for i, c in enumerate(cats)}


def apply_bin_encoder(series: pd.Series, mapping: dict) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = series.notna()
    out.loc[mask] = series.loc[mask].astype(str).map(mapping)
    return out


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    y = df["price"]
    sale_ids = df["SaleID"]
    drop_cols = ["SaleID", "price", "regDate_dt", "creatDate_dt"]
    X = df.drop(columns=drop_cols)

    X_train, X_val, y_train, y_val, _, id_val = train_test_split(
        X, y, sale_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    bin_encoders = {}
    for col in BIN_CAT_COLS:
        mapping = fit_bin_encoder(X_train[col])
        bin_encoders[col] = mapping
        X_train[col] = apply_bin_encoder(X_train[col], mapping)
        X_val[col] = apply_bin_encoder(X_val[col], mapping)

    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100,
    )

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    print(f"Train MAE:      {train_mae:,.2f}")
    print(f"Validation MAE: {val_mae:,.2f}")
    print(f"Best iteration: {model.best_iteration}")

    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    importance.tail(20).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("XGBoost feature importance (top 20)")
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "xgb_feature_importance.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(
        len(y_val), size=min(5000, len(y_val)), replace=False
    )
    ax.scatter(y_val.iloc[sample_idx], y_val_pred[sample_idx], alpha=0.25, s=10, c="teal")
    lo = min(y_val.min(), y_val_pred.min())
    hi = max(y_val.max(), y_val_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title(f"Validation: prediction vs actual (MAE={val_mae:,.0f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "xgb_pred_vs_actual.png", bbox_inches="tight")
    plt.close()

    artifact = {
        "model": model,
        "bin_encoders": bin_encoders,
        "feature_columns": X_train.columns.tolist(),
        "bin_cat_cols": BIN_CAT_COLS,
        "drop_cols": drop_cols,
        "train_mae": train_mae,
        "val_mae": val_mae,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    val_preds = pd.DataFrame(
        {
            "SaleID": id_val.values,
            "actual_price": y_val.values,
            "predicted_price": y_val_pred,
            "residual": y_val.values - y_val_pred,
            "abs_error": np.abs(y_val.values - y_val_pred),
        }
    )
    val_preds.to_csv(PRED_PATH, index=False)

    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {PRED_PATH}")
    print(f"Saved: {FIG_DIR / 'xgb_feature_importance.png'}")
    print(f"Saved: {FIG_DIR / 'xgb_pred_vs_actual.png'}")


if __name__ == "__main__":
    main()
