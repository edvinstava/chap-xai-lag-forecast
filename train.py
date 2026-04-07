import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import shap
from xgboost import XGBRegressor


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

    candidate_lags = [1, 2, 3]
    min_group_size = df.groupby('location').size().min()
    lags = [lag for lag in candidate_lags if lag < min_group_size]

    lag_source_cols = ['disease_cases']
    lag_features = []
    for lag in lags:
        for col in lag_source_cols:
            feat = f'{col}_lag_{lag}'
            df[feat] = df.groupby('location')[col].shift(lag)
            lag_features.append(feat)

    df['cases_diff_1'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.diff(1).shift(1))
    )
    df['cases_roll_mean_3'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df['cases_roll_mean_6'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    )
    df['cases_growth'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1) / (s.shift(2) + 1.0))
    )
    df['cases_per_100k'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1))
        / (df['population'] + 1.0)
    ) * 1e5

    feature_cols = [
        'rainfall',
        'mean_temperature',
        'population',
        'month_sin',
        'month_cos',
        'cases_diff_1',
        'cases_roll_mean_3',
        'cases_roll_mean_6',
        'cases_growth',
        'cases_per_100k',
    ] + lag_features
    train_df = df.dropna(subset=feature_cols + ['disease_cases'])
    test_horizon = 6
    split_idx = train_df.groupby('location').cumcount(ascending=False)
    test_mask = split_idx < test_horizon
    train_mask = ~test_mask

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, 'disease_cases']
    X_test = train_df.loc[test_mask, feature_cols]
    y_test = train_df.loc[test_mask, 'disease_cases']

    def build_model(n_rows, tree_params=None):
        if n_rows < 30:
            return LinearRegression()
        params = tree_params or {}
        return XGBRegressor(
            n_estimators=500,
            learning_rate=params.get('learning_rate', 0.03),
            max_depth=params.get('max_depth', 5),
            min_child_weight=params.get('min_child_weight', 8),
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            early_stopping_rounds=20,
        )

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    tree_grid = [
        {'learning_rate': 0.03, 'max_depth': 4, 'min_child_weight': 8},
        {'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 10},
        {'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 8},
        {'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 12},
    ]
    eval_model = build_model(len(X_train))
    best_tree_params = None
    if isinstance(eval_model, LinearRegression):
        eval_model.fit(X_train, y_train_log)
    else:
        weights = 1.0 + y_train.to_numpy() / max(float(y_train.mean()), 1.0)
        cv_time = train_df.loc[train_mask, 'time_period'].reset_index(drop=True)
        X_train_cv = X_train.reset_index(drop=True).copy()
        y_train_cv = y_train_log.reset_index(drop=True).copy()
        w_train_cv = pd.Series(weights).reset_index(drop=True)
        cv_order = cv_time.sort_values().index
        X_train_cv = X_train_cv.loc[cv_order].reset_index(drop=True)
        y_train_cv = y_train_cv.loc[cv_order].reset_index(drop=True)
        w_train_cv = w_train_cv.loc[cv_order].reset_index(drop=True)

        n_splits = 3 if len(X_train_cv) >= 40 else 2
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_rmse_log = float('inf')
        for params in tree_grid:
            fold_scores = []
            for train_idx, val_idx in tscv.split(X_train_cv):
                x_tr = X_train_cv.iloc[train_idx]
                y_tr = y_train_cv.iloc[train_idx]
                w_tr = w_train_cv.iloc[train_idx].to_numpy()
                x_val = X_train_cv.iloc[val_idx]
                y_val = y_train_cv.iloc[val_idx]

                candidate_model = build_model(len(x_tr), params)
                candidate_model.fit(
                    x_tr,
                    y_tr,
                    sample_weight=w_tr,
                    eval_set=[(x_val, y_val)],
                    verbose=False,
                )
                candidate_pred_log = candidate_model.predict(x_val)
                fold_scores.append(float(np.sqrt(mean_squared_error(y_val, candidate_pred_log))))

            candidate_rmse_log = float(np.mean(fold_scores))
            if candidate_rmse_log < best_rmse_log:
                best_rmse_log = candidate_rmse_log
                best_tree_params = params

        eval_model = build_model(len(X_train), best_tree_params)
        eval_model.fit(
            X_train,
            y_train_log,
            sample_weight=weights,
            eval_set=[(X_test, y_test_log)],
            verbose=False,
        )
    y_hat_log = eval_model.predict(X_test)
    y_hat = np.clip(np.expm1(y_hat_log), 0, None)

    y_true = y_test.to_numpy()
    nonzero = np.abs(y_true) > 1e-6
    if np.any(nonzero):
        epsilon = 1.0
        ape = np.abs((y_true[nonzero] - y_hat[nonzero]) / (y_true[nonzero] + epsilon))
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

    model = build_model(len(train_df), best_tree_params)
    y_full = train_df['disease_cases']
    y_full_log = np.log1p(y_full)
    if isinstance(model, LinearRegression):
        model.fit(train_df[feature_cols], y_full_log)
    else:
        full_weights = 1.0 + y_full.to_numpy() / max(float(y_full.mean()), 1.0)
        if len(train_df) >= 20:
            full_eval_n = max(1, int(0.1 * len(train_df)))
            model.fit(
                train_df[feature_cols].iloc[:-full_eval_n],
                y_full_log.iloc[:-full_eval_n],
                sample_weight=full_weights[:-full_eval_n],
                eval_set=[(train_df[feature_cols].iloc[-full_eval_n:], y_full_log.iloc[-full_eval_n:])],
                verbose=False,
            )
        else:
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
    print(f"Train - model: {payload['model_type']}, rows: {len(train_df)}, lags: {lags}, lag_signals: {lag_source_cols}")
    print(f"Metrics - MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, R2: {metrics['r2']:.3f}, RMSE(log): {metrics['rmse_log']:.3f}, MAPE% (nonzero): {metrics['mape_pct_nonzero_targets']:.2f}, within_20pct_accuracy (nonzero): {metrics['within_20pct_accuracy_nonzero_targets']:.3f}")
    print(f"Artifacts - metrics: {os.path.basename(metrics_fn)}, shap_plot: {os.path.basename(shap_plot_fn)}, shap_values: {os.path.basename(shap_values_fn)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a minimalist forecasting model.')

    parser.add_argument('csv_fn', type=str, help='Path to the CSV file containing input data.')
    parser.add_argument('model_fn', type=str, help='Path to save the trained model.')
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn)


