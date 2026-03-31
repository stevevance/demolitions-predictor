# Chicago Demolition Risk Predictor

An XGBoost model that estimates the probability any given Chicago parcel will be demolished within a 3-year window.

## What this does

The model trains on property features as of end-2023 and validates against actual demolitions that occurred in 2024–2025. It scores all ~934,000 active Chicago parcels and ranks them by demolition risk.

**Model performance:**

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9860 |
| PR-AUC | 0.2628 |
| Top-decile lift | 9.76x |
| Capture rate at top 5% | 93.5% (1,275 of 1,364 actual demolitions) |

At a 0.15 probability threshold, the model captures 100% of known demolitions (1,364) while flagging ~292K parcels as elevated risk.

## Features used

Ranked by mean absolute SHAP value (how much each feature moves the prediction):

| Feature | Mean |SHAP| | Description |
|---------|-------------|-------------|
| `land_ratio` | 1.063 | Land value as fraction of total assessed value |
| `building_age` | 0.765 | Age of structure in years |
| `nearby_demo_count_2yr` | 0.497 | Demolitions within ~500 ft in past 2 years |
| `renovation_investment` | 0.371 | Dollar value of recent renovation permits |
| `years_since_renovation` | 0.343 | Years since last renovation permit (99 = never) |
| `underbuilt_ratio` | 0.267 | Actual FAR / max allowed FAR under zoning |
| `is_llc_owner` | 0.219 | Owner name contains "LLC" |
| `violation_count_5yr` | 0.207 | Building code violations in past 5 years |
| Community area dummies | varies | Top 30 community areas as one-hot features |

Full SHAP values are in [`output/demolition_model_ml_importance.csv`](output/demolition_model_ml_importance.csv).

## Output files

| File | Description |
|------|-------------|
| `output/demolition_model_ml_importance.csv` | SHAP feature importances (tracked in repo) |
| `output/demolition_model_ml_top500.csv` | Top 500 highest-risk active parcels |
| `output/demolition_model_ml_validation.csv` | Known demolitions + false positives with scores |
| `output/demolition_model_ml_metrics.txt` | Full evaluation metrics |

Only the importance CSV is tracked in git. The others are large and regenerated on each run.

## Setup

```bash
pip install psycopg2-binary pandas numpy scikit-learn xgboost shap
```

Requires a PostgreSQL connection to the Chicago Cityscape database. Set environment variables or use a `.env` file:

```
DB_HOST=...
DB_PORT=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
```

Or pass a connection string directly:

```bash
python3 demolition_model_ml.py --dsn "host=... port=... dbname=... user=... password=..."
```

## Running

```bash
python3 demolition_model_ml.py
```

SHAP computation takes about a minute. To skip it on subsequent runs and reuse the saved importance CSV:

```bash
python3 demolition_model_ml.py --skip-shap
```

## Data sources

- **Cook County Assessor** — assessed values and building characteristics (2023 snapshot)
- **Chicago building permits** — renovation history and investment amounts
- **Chicago building violations** — code violations over 5-year window
- **Chicago demolition permits** — outcome variable (2024–2025 demolitions)
- **Chicago zoning** — current zoning class and max FAR

## Background

This is an upgrade from an earlier logistic regression model (`demolition_model.py`). XGBoost was chosen for improved performance on this rare-event classification task (only 0.15% of parcels are demolished in any given year).

The model was built for [Chicago Cityscape](https://www.chicagocityscape.com) to help identify properties at elevated demolition risk.
