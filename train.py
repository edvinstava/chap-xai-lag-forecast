import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap


def _load_training_dataframe(csv_fn):
    raw_df = pd.read_csv(csv_fn)

    if "Unnamed: 0" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Unnamed: 0"])

    if "population" not in raw_df.columns:
        raw_df["population"] = np.nan

    return raw_df


def train(csv_fn, model_fn):
    df = _load_training_dataframe(csv_fn)
    df['time_period'] = pd.to_datetime(df['time_period'])
    df = df.sort_values(['location', 'time_period']).reset_index(drop=True)
    df['month_sin'] = np.sin(2 * np.pi * df['time_period'].dt.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['time_period'].dt.month / 12.0)
    df['disease_cases'] = df['disease_cases'].fillna(0)
    df['population'] = pd.to_numeric(df['population'], errors='coerce')
    df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
    df['mean_temperature'] = pd.to_numeric(df['mean_temperature'], errors='coerce')

    candidate_lags = [1, 2, 3, 6, 12]
    min_group_size = df.groupby('location').size().min()
    lags = [lag for lag in candidate_lags if lag < min_group_size]

    lag_features = []
    for lag in lags:
        for col in ['rainfall', 'mean_temperature', 'population', 'disease_cases']:
            feat = f'{col}_lag_{lag}'
            df[feat] = df.groupby('location')[col].shift(lag)
            lag_features.append(feat)

    feature_cols = ['rainfall', 'mean_temperature', 'population', 'month_sin', 'month_cos'] + lag_features
    train_df = df.dropna(subset=feature_cols + ['disease_cases'])
    test_horizon = 3
    split_idx = train_df.groupby('location').cumcount(ascending=False)
    test_mask = split_idx < test_horizon
    train_mask = ~test_mask

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, 'disease_cases']
    X_test = train_df.loc[test_mask, feature_cols]
    y_test = train_df.loc[test_mask, 'disease_cases']

    def build_model(n_rows):
        if n_rows < 30:
            return LinearRegression()
        return HistGradientBoostingRegressor(
            max_iter=800,
            learning_rate=0.03,
            max_depth=5,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            max_features=0.9,
            early_stopping=False,
            random_state=42,
        )

    eval_model = build_model(len(X_train))
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    if isinstance(eval_model, LinearRegression):
        eval_model.fit(X_train, y_train_log)
    else:
        weights = 1.0 + y_train.to_numpy() / max(float(y_train.mean()), 1.0)
        eval_model.fit(X_train, y_train_log, sample_weight=weights)
    y_hat_log = eval_model.predict(X_test)
    y_hat = np.clip(np.expm1(y_hat_log), 0, None)

    y_true = y_test.to_numpy()
    nonzero = np.abs(y_true) > 1e-6
    if np.any(nonzero):
        ape = np.abs((y_true[nonzero] - y_hat[nonzero]) / y_true[nonzero])
        mape_pct = float(np.mean(ape) * 100.0)
        within_20pct_accuracy = float(np.mean(ape <= 0.20))
    else:
        mape_pct = float('nan')
        within_20pct_accuracy = float('nan')
    metrics = {
        'test_rows': int(len(y_test)),
        'mae': float(mean_absolute_error(y_test, y_hat)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_hat))),
        'r2': float(r2_score(y_test, y_hat)),
        'rmse_log': float(np.sqrt(mean_squared_error(y_test_log, y_hat_log))),
        'mape_pct_nonzero_targets': mape_pct,
        'within_20pct_accuracy_nonzero_targets': within_20pct_accuracy,
    }

    model = build_model(len(train_df))
    y_full = train_df['disease_cases']
    y_full_log = np.log1p(y_full)
    if isinstance(model, LinearRegression):
        model.fit(train_df[feature_cols], y_full_log)
    else:
        full_weights = 1.0 + y_full.to_numpy() / max(float(y_full.mean()), 1.0)
        model.fit(train_df[feature_cols], y_full_log, sample_weight=full_weights)

    shap_plot_fn = f"{model_fn}.shap_summary.png"
    shap_values_fn = f"{model_fn}.shap_values.csv"
    if not isinstance(model, LinearRegression):
        shap_sample = train_df[feature_cols].sample(n=min(300, len(train_df)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': mean_abs})
        shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
        shap_df.to_csv(shap_values_fn, index=False)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, shap_sample, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(shap_plot_fn, dpi=200)
        plt.close()
    else:
        coef_df = pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': np.abs(model.coef_)})
        coef_df = coef_df.sort_values('mean_abs_shap', ascending=False)
        coef_df.to_csv(shap_values_fn, index=False)

    payload = {
        'model': model,
        'features': feature_cols,
        'lags': lags,
        'model_type': model.__class__.__name__,
        'metrics': metrics,
        'target_transform': 'log1p'
    }
    joblib.dump(payload, model_fn)
    metrics_fn = f"{model_fn}.metrics.json"
    with open(metrics_fn, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Train - model: {payload['model_type']}, rows: {len(train_df)}, lags: {lags}")
    print(f"Metrics - MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, R2: {metrics['r2']:.3f}, RMSE(log): {metrics['rmse_log']:.3f}, MAPE% (nonzero): {metrics['mape_pct_nonzero_targets']:.2f}, within_20pct_accuracy (nonzero): {metrics['within_20pct_accuracy_nonzero_targets']:.3f}")
    print(f"Artifacts - metrics: {os.path.basename(metrics_fn)}, shap_plot: {os.path.basename(shap_plot_fn)}, shap_values: {os.path.basename(shap_values_fn)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a minimalist forecasting model.')

    parser.add_argument('csv_fn', type=str, help='Path to the CSV file containing input data.')
    parser.add_argument('model_fn', type=str, help='Path to save the trained model.')
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn)


