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
| `is_llc_owner` | 0.219 | Owner name contains "LLC" (LLC, Inc, Corp, Trust) |
| `violation_count_5yr` | 0.207 | Building code violations in past 5 years |
| `nearby_new_construction_count` | — | New construction permits within ~500 ft in past 2 years |
| `lot_size_sf` | — | Lot size in square feet |
| `sale_year` | — | Year of most recent arm's-length sale |
| `sale_price` | — | Price of most recent arm's-length sale |
| `sale_price_to_assessed_ratio` | — | Sale price ÷ total assessed value (note: class 2 residential is assessed at ~10% of market value) |
| `in_tax_sale` | — | 1 if parcel appeared as forfeited in the 2024 annual tax sale |
| `has_open_violation` | — | 1 if parcel had an open building violation at snapshot time |
| `is_vacant` | — | 1 if parcel is on the Chicago vacant building registry |
| `property_class` dummies | varies | Top 20 Cook County property classes as one-hot features |
| Community area dummies | varies | Top 30 community areas as one-hot features |

SHAP values are from the initial model run. The features marked — will be ranked after the next run incorporating the new variables. Full SHAP values are in [`output/demolition_model_ml_importance.csv`](output/demolition_model_ml_importance.csv).

## Output files

| File | Description |
|------|-------------|
| `output/demolition_model_ml_importance.csv` | SHAP feature importances |
| `output/demolition_model_ml_top500.csv` | Top 500 highest-risk active parcels |
| `output/demolition_model_ml_validation.csv` | Known demolitions + high-scoring false positives with scores |
| `output/demolition_model_ml_metrics.txt` | Full evaluation metrics |

All four files are tracked in git.

### Top 500 highest-risk parcels

[`output/demolition_model_ml_top500.csv`](output/demolition_model_ml_top500.csv) lists the 500 active Chicago parcels with the highest predicted demolition probability as of the 2023 snapshot. These are properties the model believes are most likely to be demolished by end of 2026.

Columns:

| Column | Description |
|--------|-------------|
| `pin14` | 14-digit Cook County PIN (unique parcel identifier) |
| `pin10` | 10-digit PIN (assessor-level, groups condo units) |
| `address` | Street address |
| `demolition_probability` | Model score from 0–1 (higher = more likely to be demolished) |
| `land_ratio` | Land value ÷ total assessed value (high = building is worth little relative to land) |
| `building_age` | Age of structure in years |
| `underbuilt_ratio` | Actual FAR ÷ max allowed FAR under current zoning (low = site is underbuilt) |
| `violation_count_5yr` | Building code violations filed in the past 5 years |
| `is_vacant` | 1 if parcel is on the Chicago vacant building registry |
| `in_tax_sale` | 1 if parcel appeared as forfeited in the 2024 annual tax sale |
| `community_area` | Chicago community area name |
| `zone_class` | Current zoning classification |
| `property_class` | Cook County property class code |
| `is_llc_owner` | 1 if the owner name contains LLC, Inc, Corp, or Trust |
| `nearby_demo_count_2yr` | Demolitions within ~500 ft in the past 2 years |
| `nearby_new_construction_count` | New construction permits within ~500 ft in the past 2 years |
| `years_since_renovation` | Years since last renovation permit (99 = no permit on record) |
| `renovation_investment` | Dollar value of renovation permits |
| `lot_size_sf` | Lot size in square feet |
| `sale_year` | Year of most recent arm's-length sale |
| `sale_price` | Price of most recent arm's-length sale |
| `sale_price_to_assessed_ratio` | Sale price ÷ total assessed value |

`nan` values mean the data was not available for that parcel; the model imputes medians internally before scoring.

### Validation sample

[`output/demolition_model_ml_validation.csv`](output/demolition_model_ml_validation.csv) is a diagnostic file with two groups of parcels, identified by the `sample_type` column:

- **`known_demolition`** — parcels confirmed demolished in 2024–2025, used to check that the model actually scores them highly
- **`false_positive`** — the top-scoring parcels that were *not* demolished, useful for understanding where the model is wrong

The file has the same feature columns as the top 500 list, plus:

| Column | Description |
|--------|-------------|
| `demolished_within_3yr` | 1.0 = confirmed demolished, 0.0 = not demolished |
| `has_open_violation` | 1 if the parcel had an open building violation at snapshot time |
| `sale_year` | Year of most recent arm's-length sale |
| `sale_price` | Price of most recent arm's-length sale |
| `sale_price_to_assessed_ratio` | Sale price ÷ total assessed value |
| `lot_size_sf` | Lot size in square feet |
| `nearby_new_construction_count` | New construction permits within ~500 ft in the past 2 years |
| `sample_type` | `known_demolition` or `false_positive` |

This file is most useful for spotting patterns in what the model gets wrong. For example, the current false positives are heavily concentrated in Roseland — parcels with many nearby demolitions and vacant land but no actual demo permit yet.

### Evaluation metrics

[`output/demolition_model_ml_metrics.txt`](output/demolition_model_ml_metrics.txt) contains the full model evaluation summary. Key figures from the current run (2023 snapshot, 2024–2025 outcomes):

```
Total parcels scored:     934,032
Total demolitions:          1,364  (base rate: 0.15%)

ROC-AUC:                   0.9860
PR-AUC:                    0.2628
Top-decile lift:            9.76x
Capture rate at top 5%:    93.5%  (1,275 of 1,364 demolished parcels)

Median score — demolished:     0.9082
Median score — not demolished: 0.0502

Confusion matrix at p ≥ 0.15:
  True negatives:   640,892   False positives: 291,776
  False negatives:        0   True positives:    1,364
```

**What these mean:**

- **ROC-AUC 0.986** — the model correctly ranks a random demolished parcel above a random non-demolished one 98.6% of the time
- **PR-AUC 0.263** — given how rare demolitions are (0.15% base rate), this is a strong result; random guessing would score 0.0015
- **Top-decile lift 9.76x** — the top 10% of predictions contain demolitions at nearly 10x the rate you'd expect by chance
- **Capture rate 93.5%** — 93.5% of all actual demolitions fall in the top 5% of scores
- **Zero false negatives at p ≥ 0.15** — every confirmed demolition scores above 0.15; the model misses none at this threshold (at the cost of ~292K false positives)

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

To preserve existing output files by appending a timestamp to all filenames:

```bash
python3 demolition_model_ml.py --timestamp
```

## Data sources

- **Cook County Assessor** — assessed values and building characteristics (2023 snapshot)
- **Cook County IDOR sales** — arm's-length sale year, price, and price-to-assessed ratio
- **Cook County annual tax sale** — forfeited properties from 2024 tax sale
- **Chicago building permits** — renovation history, investment amounts, and nearby new construction
- **Chicago building violations** — code violations over 5-year window
- **Chicago demolition permits** — outcome variable (2024–2025 demolitions) and nearby demolition count
- **Chicago vacant building registry** — current vacancy status
- **Chicago zoning** — current zoning class and max FAR

## Background

This is an upgrade from an earlier logistic regression model (`demolition_model.py`). XGBoost was chosen for improved performance on this rare-event classification task (only 0.15% of parcels are demolished in any given year).

The model was built for [Chicago Cityscape](https://www.chicagocityscape.com) to help identify properties at elevated demolition risk.
