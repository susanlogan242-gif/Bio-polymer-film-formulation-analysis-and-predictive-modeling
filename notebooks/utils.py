"""Shared utility functions for Materiom Take Home Assessment notebooks.

Usage in any notebook:
    from utils import *

Or selectively:
    from utils import load_data, check_data_quality, DATA_PATH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, max_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ── Constants & Config ──────────────────────────────────────────────

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "../data/materiom_dataset.csv"
CLEAN_DATA_PATH = "../data/materiom_clean.csv"

# Formulation ingredients (inputs)
INGREDIENT_COLS = [
    # TODO: Update with actual column names from dataset
    # e.g. 'chitin_%', 'clay_%', 'sorbitol_%', 'gelatin_%'
]

# Target columns (outputs)
TARGET_COLS = [
    # TODO: Update with actual column names
    # e.g. 'tensile_strength_MPa', 'optical_transmission_%'
]

# Material info reference
MATERIAL_INFO = {
    "chitin_nanofibers": "Reinforcing agent - improves tensile strength and stiffness, may reduce transparency",
    "montmorillonite_clay": "Nano-filler - improves barrier properties and fire resistance, can reduce transparency at high loadings",
    "sorbitol": "Plasticizer - improves flexibility and processability, may reduce strength",
    "gelatin": "Matrix polymer - bio-based film former, good transparency, moderate strength",
}


# ── Data Loading & Quality ──────────────────────────────────────────

def load_data(filepath=DATA_PATH):
    """Load a CSV dataset and print summary."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples, {df.shape[1]} columns")
    return df


def check_data_quality(df):
    """Print a data quality report: missing values, duplicates, numeric ranges."""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    # Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({"Count": missing, "Percentage": missing_pct})
    print(missing_df[missing_df["Count"] > 0])

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")

    # Value ranges
    print("\nNumeric Column Ranges:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            print(f"  {col}: {df[col].min():.4f} to {df[col].max():.4f}")

    return missing_df


# ── Data Cleaning ───────────────────────────────────────────────────

def clean_data(df):
    """Run the cleaning pipeline: remove duplicates, handle missing values.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Remove duplicates
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        df = df.drop_duplicates()
        print(f"Removed {n_dups} duplicate rows")

    print(f"Clean dataset: {len(df)} samples")
    return df


# ── Formulation Analysis (Task 1) ──────────────────────────────────

def normalise_column(series):
    """Normalise a Series to 0-1 range."""
    return (series - series.min()) / (series.max() - series.min())


def score_formulations(df, criteria, weights=None):
    """Score formulations based on multiple criteria.

    Args:
        df: DataFrame with property columns.
        criteria: dict of {column_name: 'higher' or 'lower'} indicating desired direction.
        weights: dict of {column_name: weight}. If None, equal weights used.

    Returns:
        Series of scores (0-1, higher = better).
    """
    if weights is None:
        weights = {col: 1.0 / len(criteria) for col in criteria}

    score = pd.Series(0.0, index=df.index)
    for col, direction in criteria.items():
        normed = normalise_column(df[col])
        if direction == "lower":
            normed = 1 - normed
        score += weights.get(col, 0) * normed

    return score


# ── Correlation & Analysis ──────────────────────────────────────────

def get_correlation_matrix(df, cols=None):
    """Compute correlation matrix for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    return df[cols].corr()


# ── Visualization ───────────────────────────────────────────────────

def plot_correlation_heatmap(df, cols=None, cmap="RdYlGn", save_path=None):
    """Plot a correlation heatmap."""
    corr = get_correlation_matrix(df, cols)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                center=0, square=True, linewidths=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_scatter(df, x_col, y_col, color="steelblue", save_path=None):
    """Simple scatter plot between two columns."""
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.6, color=color)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """Scatter plot of actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color="steelblue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """Bar plot of feature importances from a tree-based model."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="teal")
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    return importances


# ── Model Evaluation ────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, label=""):
    """Print MAE, Max Error, and R2 for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    me = max_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  Max Error: {me:.4f}")
    print(f"  R2:        {r2:.4f}")
    return {"MAE": mae, "Max Error": me, "R2": r2}


# ── Quick Load ──────────────────────────────────────────────────────

def load_clean_data(filepath=CLEAN_DATA_PATH):
    """Load the already-cleaned dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded clean data: {len(df)} rows, {df.shape[1]} columns")
    return df
