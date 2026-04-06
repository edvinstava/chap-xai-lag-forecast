# chap-xai-lag-forecast

An example of an **external Python model** compatible with [CHAP](https://dhis2-chap.github.io/chap-core/). It mirrors the structure of community examples such as [chap_auto_ewars](https://github.com/chap-models/chap_auto_ewars): a single `MLproject` file defines how CHAP invokes training and prediction, and small scripts implement the logic.

The model uses lagged covariates and calendar features, fits either **linear regression** (small samples) or **XGBoost** (larger samples) on a **log1p** target, and can emit **native SHAP** values for explainability when CHAP’s XAI integration is enabled.

## Naming

| What | Suggested value |
|------|------------------|
| **GitHub repository** | `chap-xai-lag-forecast` (or `chap_xai_lag_forecast` if you prefer underscores) |
| **Model id** (`name` in `MLproject`) | `chap_xai_lag_forecast` |

After cloning, replace `/path/to/chap-xai-lag-forecast` below with your local directory (`pwd`).

## How CHAP runs this model

CHAP reads `MLproject` (the analogue of an older `config.yml` + train/predict commands in some R examples). It is equivalent in spirit to:

```yaml
name: chap_xai_lag_forecast
train_command: 'python train.py {train_data} {model}'
predict_command: 'python predict.py {model} {historic_data} {future_data} {out_file}'
```

There is **no adapter map** in this Python example: train and predict already use CHAP’s canonical column names (`time_period`, `location`, `rainfall`, `mean_temperature`, `population`, `disease_cases`). CHAP writes harmonised CSVs with those fields; this repository does not ship a separate DHIS2-wide converter.

The actual `MLproject` in this repo:

```yaml
name: chap_xai_lag_forecast

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"
```

- **Train** runs `python train.py` with the path to the training CSV CHAP produced and the path where the fitted artefact should be saved.
- **Predict** runs `python predict.py` with the saved model, historic context CSV, future covariate CSV, and the output predictions path.

## Data CHAP passes in

### `train_data`

Harmonised long format, with at least:

- `time_period` — month (CHAP typically uses `YYYY-MM`)
- `location` — org unit / region label
- `rainfall`, `mean_temperature` — climate covariates
- `population` — optional but used when present
- `disease_cases` — target (may contain gaps; zeros used where needed for training)

An index column such as `Unnamed: 0` is ignored if present.

### `historic_data` and `future_data` (predict)

- **Historic** data: same schema as training for the window CHAP uses as context (lags and state are updated per location).
- **Future** data: same covariate columns; `disease_cases` is typically empty for rows to forecast. The model writes point predictions into `sample_0` on the output rows.

### `model`

A `joblib` payload: fitted estimator, feature list, lag configuration, and metadata (see `train.py`). Not intended for hand-editing.

### `out_file`

CSV of predictions (including `sample_0`). CHAP consumes this for evaluation and reporting.

## Explainability (SHAP) and CHAP

Training optionally writes:

- `{model}.shap_summary.png` — summary plot (tree models only)
- `{model}.shap_values.csv` — aggregated feature importance table

During **predict**, when the payload is the full dict format, the script writes `shap_values.csv` in the run working directory with per-row SHAP contributions and `expected_value` (TreeExplainer for XGBoost; a linear fallback for `LinearRegression`).

CHAP can surface **native** SHAP-style outputs from external models when you enable the corresponding integration:

```bash
CHAP_FORCE_NATIVE_SHAP=true chap evaluate \
  --model-name /path/to/chap-xai-lag-forecast \
  --dataset-name ISIMIP_dengue_harmonized \
  --dataset-country brazil \
  --report-filename report.pdf \
  --ignore-environment \
  --debug
```

Use a CHAP Core version that supports native SHAP for external models; the flag opts into that path so evaluation and UI can attach XAI to your run.

## Local development

```bash
cd /path/to/chap-xai-lag-forecast
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install chap-core
```

Smoke test with the tiny CSVs under `input/` (if present in your clone):

```bash
python train.py input/trainData.csv output/model.bin
python predict.py output/model.bin input/trainData.csv input/futureClimateData.csv output/predictions.csv
```

`example_data/training_data.csv` is **git-ignored** by default: keep a copy locally for heavier tests without committing data.

## Publish to GitHub

Create an empty repository on GitHub named e.g. `chap-xai-lag-forecast`, then:

```bash
cd /path/to/chap-xai-lag-forecast
git init
git add MLproject train.py predict.py requirements.txt README.md .gitignore
git add input/ 2>/dev/null || true   # optional small fixtures only if you track them
git commit -m "Initial CHAP external model: lag forecast with XAI hooks"
git branch -M main
git remote add origin https://github.com/YOUR_USER/chap-xai-lag-forecast.git
git push -u origin main
```

Replace `YOUR_USER` and the URL if you use SSH:

```bash
git remote add origin git@github.com:YOUR_USER/chap-xai-lag-forecast.git
git push -u origin main
```

Evaluate from a clone:

```bash
CHAP_FORCE_NATIVE_SHAP=true chap evaluate \
  --model-name /path/to/chap-xai-lag-forecast \
  --dataset-name ISIMIP_dengue_harmonized \
  --dataset-country brazil \
  --report-filename report.pdf \
  --ignore-environment \
  --debug
```

## Register in a local CHAP backend (optional)

If your stack loads configured models from the database, point `source_url` at the **local clone path** or **GitHub URL** of this repo, then add a default configuration; restart `chap serve`. See CHAP Core docs for your deployment’s seeding or admin UI.

---

Reference R-style external model documentation for comparison: [chap-models/chap_auto_ewars](https://github.com/chap-models/chap_auto_ewars).
