
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.model_selection import cross_val_score


# Data Loading

def load_data(filepath="data/winequality-red.csv"):
    """
    Load the wine dataset.

    Parameters
    ----------
    filepath : str
        Path to dataset.

    Returns
    -------
    DataFrame
    """
    df = pd.read_csv(filepath)
    return df


# Data Cleaning


def binarize_quality(df, threshold=7):
    """
    Convert quality score to binary target.

    quality >= threshold → 1 (good wine)
    quality < threshold  → 0
    """
    data = df.copy()

    data["good_quality"] = (
        data["quality"] >= threshold
    ).astype(int)

    return data


def drop_quality(df):
    """Remove original integer quality column."""
    return df.drop(columns=["quality"], errors="ignore")


# Feature Engineering

def create_features(df):
    """
    Create chemistry-based derived features.
    """

    data = df.copy()

    # Total acidity
    data["total_acidity"] = (
        data["fixed_acidity"] + data["volatile_acidity"]
    )

    # Ratio: fixed / volatile acidity
    data["acidity_ratio"] = (
        data["fixed_acidity"] /
        data["volatile_acidity"].replace(0, np.nan)
    )

    # Ratio: free sulfur / total sulfur
    data["free_to_total_sulfur"] = (
        data["free_sulfur_dioxide"] /
        data["total_sulfur_dioxide"].replace(0, np.nan)
    )

    # Alcohol bins
    data["alcohol_level"] = pd.cut(
        data["alcohol"],
        bins=[0, 9.5, 11, 14],
        labels=["low", "medium", "high"]
    )

    return data


def preprocess_data(df, threshold=7):
    """
    Complete preprocessing pipeline.

    Steps
  
    1. Binarize quality
    2. Drop original quality
    3. Feature engineering
    4. One-hot encoding
    5. Handle NaNs
    """

    data = binarize_quality(df, threshold)

    data = drop_quality(data)

    data = create_features(data)

    # One‑hot encoding
    data = pd.get_dummies(
        data,
        columns=["alcohol_level"],
        drop_first=True
    )

    # Fill NaNs created by ratios
    data = data.fillna(
        data.median(numeric_only=True)
    )

    return data


# Model Evaluation

def evaluate_model(model_name, y_true, y_pred):
    """
    Compute and print classification metrics.
    """

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }

    print("\n" + "=" * 40)
    print(f"{model_name}")
    print("=" * 40)

    for k, v in metrics.items():
        if k != "Model":
            print(f"{k:10s}: {v:.4f}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, ax=None):
    """
    Plot confusion matrix heatmap.
    """

    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Good", "Good"],
        yticklabels=["Not Good", "Good"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(f"Confusion Matrix — {model_name}")

    return ax


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Plot ROC curves for multiple models.
    """

    plt.figure(figsize=(10,7))

    for name, model in models_dict.items():

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:,1]

        elif hasattr(model, "decision_function"):
            probs = model.decision_function(X_test)

        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, probs)

        model_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={model_auc:.3f})"
        )

    plt.plot([0,1],[0,1],"k--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve Comparison")

    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def cross_validate_model(model, X, y, cv=5):
    """
    Run cross‑validation using F1 score.
    """

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1"
    )

    print(f"CV F1 Scores : {scores.round(4)}")
    print(f"Mean F1      : {scores.mean():.4f} ± {scores.std():.4f}")

    return scores


def compare_models(results_list):
    """
    Build comparison table for multiple models.
    """

    df_results = pd.DataFrame(results_list)

    df_results = (
        df_results
        .sort_values("F1 Score", ascending=False)
        .reset_index(drop=True)
    )

    return df_results
