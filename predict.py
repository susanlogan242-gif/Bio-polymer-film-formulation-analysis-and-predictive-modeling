"""
Prediction entrypoint for BioFilm assessment.

Usage:
    python predict.py --input data/holdout_dataset.csv --output results/predictions.csv

This loads the trained models and predicts tensile strength and optical transmission
for a given dataset.
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.special import inv_boxcox
from sklearn.metrics import mean_absolute_error, max_error


def load_models(model_dir="models"):
    model_tensile = joblib.load(f"{model_dir}/model_tensile.pkl")
    model_optical = joblib.load(f"{model_dir}/model_optical.pkl")
    return model_tensile, model_optical


def inverse_transform_optical(pred, transform_info):
    """Convert optical predictions from transformed space back to original scale."""
    if transform_info is None or transform_info.get('adopted') is None:
        return pred
    name = transform_info['adopted']
    if name == 'Log (reflected)':
        return 101 - np.exp(pred)
    elif name == 'Square':
        return np.sqrt(np.clip(pred, 0, None))
    elif name == 'Box-Cox':
        return inv_boxcox(pred, transform_info['bc_lambda'])
    return pred


def predict(df, model_tensile, model_optical, feature_cols, scaler=None, transform_info=None):
    X = df[feature_cols].copy()
    X = X.fillna(X.mean())
    if scaler is not None:
        X = scaler.transform(X)

    pred_tensile = model_tensile.predict(X)
    pred_optical_raw = model_optical.predict(X)
    pred_optical = inverse_transform_optical(pred_optical_raw, transform_info)

    return pred_tensile, pred_optical


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    me = max_error(y_true, y_pred)
    print(f"{label} - MAE: {mae:.3f}, Max Error: {me:.3f}")
    return mae, me


def main():
    parser = argparse.ArgumentParser(description="Predict tensile strength and optical transmission")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="results/predictions.csv", help="Path to save predictions")
    parser.add_argument("--models", default="models", help="Directory containing trained models")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples from {args.input}")

    # Load models and feature list
    model_tensile, model_optical = load_models(args.models)
    feature_cols = joblib.load(f"{args.models}/feature_cols.pkl")

    # Engineer features (must match notebook 03)
    ingredient_cols = ['sorbitol', 'montmorillonite_clay', 'gelatin', 'chitin_nanofibers']
    df['chitin_gelatin_ratio'] = df['chitin_nanofibers'] / (df['gelatin'] + 1e-6)
    df['total_reinforcement'] = df['chitin_nanofibers'] + df['montmorillonite_clay']
    df['chitin_x_clay'] = df['chitin_nanofibers'] * df['montmorillonite_clay']
    df['gelatin_x_sorbitol'] = df['gelatin'] * df['sorbitol']

    # Scale features
    scaler = joblib.load(f"{args.models}/scaler.pkl")

    # Load optical transform info (if any)
    try:
        transform_info = joblib.load(f"{args.models}/optical_transform.pkl")
    except FileNotFoundError:
        transform_info = None

    # Predict
    pred_tensile, pred_optical = predict(df, model_tensile, model_optical, feature_cols, scaler, transform_info)

    # Save predictions
    results = df.copy()
    results["pred_tensile_strength"] = pred_tensile
    results["pred_optical_transmission"] = pred_optical
    results.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    # If ground truth columns exist, evaluate
    # TODO: Update target column names
    # if 'tensile_strength_col' in df.columns:
    #     evaluate(df['tensile_strength_col'], pred_tensile, "Tensile Strength")
    # if 'optical_transmission_col' in df.columns:
    #     evaluate(df['optical_transmission_col'], pred_optical, "Optical Transmission")


if __name__ == "__main__":
    main()
