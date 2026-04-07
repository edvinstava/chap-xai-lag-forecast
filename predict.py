import argparse

import joblib
import numpy as np
import pandas as pd
import shap


def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    def write_native_shap(model_obj, x_df, out_df):
        if x_df.empty:
            return
        try:
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(x_df)
            expected_value = explainer.expected_value
            if np.isscalar(expected_value):
                expected = np.repeat(float(expected_value), len(x_df))
            else:
                expected = np.repeat(float(np.array(expected_value).reshape(-1)[0]), len(x_df))
        except Exception:
            if not hasattr(model_obj, "coef_") or not hasattr(model_obj, "intercept_"):
                return
            background = x_df.mean()
            shap_values = (x_df - background).to_numpy() * np.array(model_obj.coef_)
            expected = np.repeat(float(model_obj.intercept_ + np.dot(background.values, model_obj.coef_)), len(x_df))

        shap_df = pd.DataFrame(shap_values, columns=x_df.columns)
        shap_df.insert(0, 'time_period', out_df['time_period'].values)
        shap_df.insert(0, 'location', out_df['location'].values)
        shap_df.insert(2, 'expected_value', expected)
        shap_df.to_csv("shap_values.csv", index=False)

    payload = joblib.load(model_fn)
    future_df = pd.read_csv(future_climatedata_fn)
    historic_df = pd.read_csv(historic_data_fn)

    if not isinstance(payload, dict) or 'model' not in payload or 'features' not in payload:
        if 'population' not in future_df.columns:
            future_df['population'] = 0.0
        X = future_df[['rainfall', 'mean_temperature', 'population']]
        y_pred = payload.predict(X)
        future_df['sample_0'] = y_pred
        shap_out_df = future_df[['location', 'time_period']].copy()
        shap_out_df['time_period'] = pd.to_datetime(shap_out_df['time_period']).dt.strftime('%Y%m_1')
        write_native_shap(payload, X, shap_out_df)
        future_df.to_csv(predictions_fn, index=False)
        print("Predictions: ", y_pred)
        return y_pred

    model = payload['model']
    feature_cols = payload['features']
    lags = payload.get('lags', [])
    target_transform = payload.get('target_transform')

    historic_df['time_period'] = pd.to_datetime(historic_df['time_period'])
    future_df['time_period'] = pd.to_datetime(future_df['time_period'])
    historic_df = historic_df.sort_values(['location', 'time_period']).reset_index(drop=True)
    future_df = future_df.sort_values(['location', 'time_period']).reset_index(drop=True)
    if 'population' not in historic_df.columns:
        historic_df['population'] = np.nan
    historic_df['population'] = pd.to_numeric(historic_df['population'], errors='coerce')
    historic_df['rainfall'] = pd.to_numeric(historic_df['rainfall'], errors='coerce')
    historic_df['mean_temperature'] = pd.to_numeric(historic_df['mean_temperature'], errors='coerce')
    if 'disease_cases' not in historic_df.columns:
        historic_df['disease_cases'] = 0.0
    historic_df['disease_cases'] = pd.to_numeric(historic_df['disease_cases'], errors='coerce').fillna(0.0)
    if 'population' not in future_df.columns:
        fallback_population = historic_df['population'].mean() if len(historic_df) else 0.0
        future_df['population'] = fallback_population
    future_df['population'] = pd.to_numeric(future_df['population'], errors='coerce')

    global_defaults = {
        'rainfall': historic_df['rainfall'].mean() if len(historic_df) else 0.0,
        'mean_temperature': historic_df['mean_temperature'].mean() if len(historic_df) else 0.0,
        'population': historic_df['population'].mean() if len(historic_df) else 0.0,
        'disease_cases': historic_df['disease_cases'].mean() if len(historic_df) else 0.0,
    }
    by_loc_defaults = (
        historic_df.groupby('location')[['rainfall', 'mean_temperature', 'population', 'disease_cases']]
        .mean()
        .to_dict(orient='index')
    )

    state_by_loc = {
        loc: grp[['time_period', 'rainfall', 'mean_temperature', 'population', 'disease_cases']].copy().reset_index(drop=True)
        for loc, grp in historic_df.groupby('location')
    }

    x_rows = []
    preds = []
    for idx, row in future_df.iterrows():
        loc = row['location']
        defaults = by_loc_defaults.get(loc, global_defaults)
        if loc not in state_by_loc:
            state_by_loc[loc] = pd.DataFrame(
                columns=['time_period', 'rainfall', 'mean_temperature', 'population', 'disease_cases']
            )

        state = state_by_loc[loc]
        feat = {
            'rainfall': row['rainfall'],
            'mean_temperature': row['mean_temperature'],
            'month_sin': np.sin(2 * np.pi * row['time_period'].month / 12.0),
            'month_cos': np.cos(2 * np.pi * row['time_period'].month / 12.0)
        }

        for lag in lags:
            for col in ['rainfall', 'mean_temperature', 'population', 'disease_cases']:
                key = f'{col}_lag_{lag}'
                if len(state) >= lag:
                    feat[key] = state.iloc[-lag][col]
                else:
                    feat[key] = defaults[col]

        x = pd.DataFrame([{c: feat.get(c, 0.0) for c in feature_cols}])
        x_rows.append(x.iloc[0].to_dict())
        y_raw = float(model.predict(x)[0])
        if target_transform == 'log1p':
            y_hat = float(np.clip(np.expm1(y_raw), 0, None))
        else:
            y_hat = y_raw
        preds.append(y_hat)
        future_df.at[idx, 'sample_0'] = y_hat

        new_row = pd.DataFrame([{
            'time_period': row['time_period'],
            'rainfall': row['rainfall'],
            'mean_temperature': row['mean_temperature'],
            'population': row['population'],
            'disease_cases': y_hat
        }])
        state_by_loc[loc] = pd.concat([state, new_row], ignore_index=True)

    future_df['time_period'] = future_df['time_period'].dt.strftime('%Y-%m')
    x_pred_df = pd.DataFrame(x_rows)[feature_cols]
    shap_out_df = future_df[['location', 'time_period']].copy()
    write_native_shap(model, x_pred_df, shap_out_df)
    future_df.to_csv(predictions_fn, index=False)
    print("Predictions: ", preds)
    return np.array(preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using the trained model.')

    parser.add_argument('model_fn', type=str, help='Path to the trained model file.')
    parser.add_argument('historic_data_fn', type=str, help='Path to the CSV file historic data (here ignored).')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to the CSV file containing future climate data.')
    parser.add_argument('predictions_fn', type=str, help='Path to save the predictions CSV file.')

    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)
