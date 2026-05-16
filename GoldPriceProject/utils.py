import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score


def load_data(filepath="data/gold_price_data.csv"):
    # Read the dataset
    df = pd.read_csv(filepath)

    # Convert Date column to datetime so we can work with it easily
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by date to keep the time order correct
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def create_features(df):
    # Work on a copy so we don't accidentally modify the original dataframe
    df = df.copy()

    # Extract some simple time related features
    if "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Create a few basic rolling and difference features for market variables
    for col in ["SPX", "USO", "SLV", "EUR/USD"]:
        if col in df.columns:
            # 5‑day rolling mean to capture short-term trend
            df[f"{col}_roll5"] = df[col].rolling(5, min_periods=1).mean()

            # Day-to-day change
            df[f"{col}_diff"] = df[col].diff().fillna(0)

    # Ratio between silver ETF and oil ETF (sometimes useful for macro signals)
    if {"SLV", "USO"}.issubset(df.columns):
        df["SLV_USO_ratio"] = df["SLV"] / df["USO"].replace(0, np.nan)

    # Drop the raw date column since models can't use it directly
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # Fill missing values so models don't crash
    df = df.fillna(method="bfill").fillna(0)

    return df


def preprocess_data(df):
    # Run feature engineering first
    df = create_features(df)

    # Convert boolean columns to 0/1 if there are any
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def evaluate_model(model_name, y_true, y_pred):
    # Calculate RMSE separately just for readability
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {
        "Model": model_name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }

    print(f"\n{'='*40}\n  {model_name}\n{'='*40}")

    # Print all metrics except the model name
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:6s}: {v:.4f}")

    return metrics


def plot_pred_vs_actual(y_true, y_pred, model_name, ax=None):
    # Create a figure if an axis was not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Scatter plot of predictions vs real values
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")

    # Draw a perfect prediction line (y = x)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual — {model_name}")
    ax.legend()

    return ax


def plot_residuals(y_true, y_pred, model_name):
    # Two plots: residual scatter + residual distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    residuals = np.array(y_true) - np.array(y_pred)

    # Residuals vs predictions
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs Predicted — {model_name}")

    # Histogram of residuals
    axes[1].hist(residuals, bins=40, color="seagreen", edgecolor="black")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual distribution")

    plt.tight_layout()
    return fig


def cross_validate_model(model, X, y, cv=5):
    # Standard k-fold cross validation using R²
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    print(f"  CV R² Scores : {scores.round(4)}")
    print(f"  Mean R²      : {scores.mean():.4f} (+/- {scores.std():.4f})")

    return scores


def compare_models(results_list):
    # Turn list of metric dictionaries into a dataframe
    df_results = pd.DataFrame(results_list)

    # Sort by R² so the best model appears first
    return df_results.sort_values("R2", ascending=False).reset_index(drop=True)
