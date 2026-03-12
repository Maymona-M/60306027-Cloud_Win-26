import argparse
import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Azure
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import re

def clean_col_name(col):
    col = re.sub(r'["\(\)]', '', col)
    col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
    col = re.sub(r'_+', '_', col)
    return col.strip('_')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--output_data', type=str)
    args = parser.parse_args()

    # Load
    X = pd.read_csv(os.path.join(args.input_data, 'features_final.csv'))
    y = pd.read_csv(os.path.join(args.input_data, 'rul_labels.csv')).squeeze()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Clean column names for LightGBM
    X.columns = [clean_col_name(c) for c in X.columns]

    # Define models
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost':           XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM':          LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }

    # Train and evaluate
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y,
                                 cv=kf,
                                 scoring='neg_root_mean_squared_error')
        rmse = -scores.mean()
        std = scores.std()
        results[name] = {'RMSE': round(rmse, 2), 'STD': round(std, 2)}
        print(f"{name:25s} → RMSE: {rmse:.2f} (±{std:.2f})")

    # Find best model
    best_name = min(results, key=lambda x: results[x]['RMSE'])
    print(f"\nBest model: {best_name} → RMSE: {results[best_name]['RMSE']}")

    # Train best model on full data
    best_model = models[best_name]
    best_model.fit(X, y)
    y_pred = best_model.predict(X)

    # Save outputs
    os.makedirs(args.output_data, exist_ok=True)

    # Save model
    joblib.dump(best_model, os.path.join(args.output_data, 'best_model.pkl'))

    # Save metrics
    final_metrics = {
        'best_model': best_name,
        'cv_rmse': results[best_name]['RMSE'],
        'cv_std': results[best_name]['STD'],
        'n_features_used': X.shape[1],
        'all_models': results
    }
    with open(os.path.join(args.output_data, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Plot model comparison
    names = list(results.keys())
    rmses = [results[m]['RMSE'] for m in names]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(names, rmses, color=['steelblue', 'orange', 'green', 'red'])
    plt.ylabel('RMSE (cycles)')
    plt.title('Model Comparison — RUL Prediction')
    plt.ylim(0, max(rmses) * 1.3)
    for bar, val in zip(bars, rmses):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 str(val), ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_data, 'model_comparison.png'))
    plt.close()

    # Plot predicted vs actual
    plt.figure(figsize=(8, 5))
    plt.scatter(y, y_pred, alpha=0.7, color='steelblue', edgecolors='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()],
             'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel('Actual RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')
    plt.title(f'Predicted vs Actual RUL — {best_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_data, 'predicted_vs_actual.png'))
    plt.close()

    print("Train and evaluate complete.")

if __name__ == '__main__':
    main()