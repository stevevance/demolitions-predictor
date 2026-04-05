# Chicago Demolition Risk Predictor

An XGBoost model that estimates the probability any given Chicago parcel will be demolished within a 3-year window.

## What this does

The model trains on property features as of end-2023 and validates against actual demolitions that occurred in 2024–2026. It scores 480,600 active Chicago parcels — residential properties and vacant land, excluding condos — and ranks them by demolition risk.

**Model performance:**

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9831 |
| PR-AUC | 0.1588 |
| Top-decile lift | 9.77x |
| Capture rate at top 5% | 91.4% (994 of 1,087 actual demolitions) |

At a 0.15 probability threshold, the model captures 100% of known demolitions (1,087) while flagging ~127K parcels as elevated risk.

## Features used

Ranked by mean absolute SHAP value (how much each feature moves the prediction):

| Feature | Mean Abs SHAP | Description |
|---------|--------------|-------------|
| `building_age` | 0.758 | Age of structure in years |
| `nearby_demo_count_2yr` | 0.520 | Demolitions within ~500 ft in past 2 years |
| `land_ratio` | 0.349 | Land value as fraction of total assessed value |
| `underbuilt_ratio` | 0.296 | Actual FAR / max allowed FAR under zoning |
| `land_val` | 0.281 | Assessed land value (2023 snapshot) |
| `lot_size_sf` | 0.272 | Lot size in square feet |
| `years_since_renovation` | 0.236 | Years since last renovation permit (99 = never) |
| `nearby_new_construction_count` | 0.222 | New construction permits within ~500 ft in past 2 years |
| `land_val_change_pct` | 0.216 | % change in assessed land value, 2018→2023 |
| `building_val` | 0.198 | Assessed building value (2023 snapshot) |
| `in_tax_sale` | 0.184 | 1 if parcel appeared as forfeited in the 2024 annual tax sale |
| `ew_Masonry` | 0.183 | Exterior wall construction type = Masonry |
| `violation_count_5yr` | 0.163 | Building code violations in past 5 years |
| `is_llc_owner` | 0.162 | Owner name contains LLC, Inc, Corp, or Trust |
| `building_val_change_pct` | 0.158 | % change in assessed building value, 2018→2023 |
| `sale_price_to_assessed_ratio` | 0.142 | Sale price ÷ total assessed value |
| `renovation_investment` | 0.100 | Dollar value of recent renovation permits |
| `sale_year` | 0.100 | Year of most recent arm's-length sale |
| `sale_price` | 0.091 | Price of most recent arm's-length sale |
| `is_vacant` | 0.077 | 1 if parcel is on the Chicago vacant building registry |
| `ew_Frame` | 0.074 | Exterior wall construction type = Frame |
| `has_open_violation` | 0.053 | 1 if parcel had an open building violation at snapshot time |
| `property_class` dummies | varies | Top 20 Cook County property classes as one-hot features |
| Community area dummies | varies | Top 30 community areas as one-hot features |
| `char_ext_wall` dummies | varies | Exterior wall construction type as one-hot features |

Full SHAP values are in [`output/demolition_model_ml_importance.csv`](output/demolition_model_ml_importance.csv).

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
| `land_val` | Assessed land value (2023) |
| `building_val` | Assessed building value (2023) |
| `land_val_change_pct` | % change in assessed land value from 2018 to 2023 |
| `building_val_change_pct` | % change in assessed building value from 2018 to 2023 |
| `building_age` | Age of structure in years |
| `underbuilt_ratio` | Actual FAR ÷ max allowed FAR under current zoning (low = site is underbuilt) |
| `violation_count_5yr` | Building code violations filed in the past 5 years |
| `is_vacant` | 1 if parcel is on the Chicago vacant building registry |
| `in_tax_sale` | 1 if parcel appeared as forfeited in the 2024 annual tax sale |
| `community_area` | Chicago community area name |
| `zone_class` | Current zoning classification |
| `property_class` | Cook County property class code |
| `char_ext_wall` | Exterior wall construction type (Frame, Masonry, Frame + Masonry, Stucco) |
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

- **`known_demolition`** — parcels confirmed demolished in 2024–2026, used to check that the model actually scores them highly
- **`false_positive`** — the top-scoring parcels that were *not* demolished, useful for understanding where the model is wrong

The file has the same feature columns as the top 500 list, plus:

| Column | Description |
|--------|-------------|
| `demolished_within_3yr` | 1.0 = confirmed demolished, 0.0 = not demolished |
| `has_open_violation` | 1 if the parcel had an open building violation at snapshot time |
| `sample_type` | `known_demolition` or `false_positive` |

### Evaluation metrics

[`output/demolition_model_ml_metrics.txt`](output/demolition_model_ml_metrics.txt) contains the full model evaluation summary. Key figures from the current run (2023 snapshot, 2024–2026 outcomes):

```
Total parcels scored:     480,600
Total demolitions:          1,087  (base rate: 0.23%)

ROC-AUC:                   0.9831
PR-AUC:                    0.1588
Top-decile lift:            9.77x
Capture rate at top 5%:    91.4%  (994 of 1,087 demolished parcels)

Median score — demolished:     0.9181
Median score — not demolished: 0.0497

Confusion matrix at p ≥ 0.15:
  True negatives:   352,566   False positives: 126,947
  False negatives:        0   True positives:    1,087
```

**What these mean:**

- **ROC-AUC 0.983** — the model correctly ranks a random demolished parcel above a random non-demolished one 98.3% of the time
- **PR-AUC 0.159** — given how rare demolitions are (0.23% base rate), this is a strong result; random guessing would score 0.0023
- **Top-decile lift 9.77x** — the top 10% of predictions contain demolitions at nearly 10x the rate you'd expect by chance
- **Capture rate 91.4%** — 91.4% of all actual demolitions fall in the top 5% of scores
- **Zero false negatives at p ≥ 0.15** — every confirmed demolition scores above 0.15; the model misses none at this threshold (at the cost of ~127K false positives)

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

To run hyperparameter tuning with RandomizedSearchCV before training:

```bash
python3 demolition_model_ml.py --tune
```

## Data sources

- **Cook County Assessor** — assessed values and building characteristics (2023 snapshot); 2018 values for change features
- **Cook County IDOR sales** — arm's-length sale year, price, and price-to-assessed ratio
- **Cook County annual tax sale** — forfeited properties from 2024 tax sale
- **Chicago building permits** — renovation history, investment amounts, and nearby new construction
- **Chicago building violations** — code violations over 5-year window
- **Chicago demolition permits** — outcome variable (2024–2026 demolitions) and nearby demolition count
- **Chicago vacant building registry** — current vacancy status
- **Chicago zoning** — current zoning class and max FAR

## What the results show

The model was trained and validated on a 2023 snapshot against actual demolitions in 2024–2026. Here's what the outputs reveal.

**Top 10 highest-risk active parcels (2023 snapshot):**

| Address | Community Area | Zoning | Probability | Building Age | LLC-owned | Nearby Demos |
|---------|---------------|--------|-------------|-------------|-----------|-------------|
| 3048 N Clybourn Ave | North Center | C1-2 | 99.8% | 130 yrs | Yes | 15 |
| 1221 W Grand Ave | West Town | M2-2 | 99.8% | 133 yrs | Yes | 11 |
| 2344 W Lyndale St | Logan Square | RT-4 | 99.8% | 135 yrs | Yes | 10 |
| 2138 W Barry Ave | North Center | RS-3 | 99.7% | 136 yrs | Yes | 28 |
| 1802 N Cleveland Ave | Lincoln Park | RM-5 | 99.7% | 139 yrs | Yes | 21 |
| 1951 N Burling St | Lincoln Park | RM-4.5 | 99.7% | 143 yrs | Yes | 34 |
| 1829 N Talman Ave | Logan Square | RS-3 | 99.7% | 133 yrs | No | 6 |
| 3318 N Clifton Ave | Lake View | RT-4 | 99.7% | 135 yrs | Yes | 18 |
| 2032 W Chicago Ave | West Town | B3-2 | 99.7% | 134 yrs | Yes | 20 |
| 1938 W Eddy St | North Center | RS-3 | 99.7% | 119 yrs | Yes | 24 |

Nine of the ten are LLC-owned. All are frame or masonry construction, 119–143 years old, with active surrounding demolition activity. This is the classic North Side teardown profile — old buildings on commercially or residentially zoned land that has appreciated dramatically, where a developer's value is in the lot, not the structure.

The three community areas with the most parcels in the top 500 are Lake View (84), North Center (78), and Lincoln Park (62). Their top 5 highest-risk parcels:

**Lake View**

| Address | Zoning | Probability | Building Age | LLC-owned | Nearby Demos |
|---------|--------|-------------|-------------|-----------|-------------|
| 3318 N Clifton Ave | RT-4 | 99.7% | 135 yrs | Yes | 18 |
| 872 W Buckingham Pl | RM-5 | 99.6% | 127 yrs | Yes | 10 |
| 1752 W Newport Ave | RS-3 | 99.4% | 67 yrs | Yes | 21 |
| 708 W Briar Pl | RM-4.5 | 99.4% | 135 yrs | Yes | 8 |
| 3755 N Greenview Ave | RT-3.5 | 99.3% | 133 yrs | Yes | 8 |

**North Center**

| Address | Zoning | Probability | Building Age | LLC-owned | Nearby Demos |
|---------|--------|-------------|-------------|-----------|-------------|
| 3048 N Clybourn Ave | C1-2 | 99.8% | 130 yrs | Yes | 15 |
| 2138 W Barry Ave | RS-3 | 99.7% | 136 yrs | Yes | 28 |
| 1938 W Eddy St | RS-3 | 99.7% | 119 yrs | Yes | 24 |
| 1925 W Cornelia Ave | RS-3 | 99.6% | 121 yrs | Yes | 27 |
| 3238 N Wolcott Ave | RS-3 | 99.4% | 119 yrs | Yes | 17 |

**Lincoln Park**

| Address | Zoning | Probability | Building Age | LLC-owned | Nearby Demos |
|---------|--------|-------------|-------------|-----------|-------------|
| 1802 N Cleveland Ave | RM-5 | 99.7% | 139 yrs | Yes | 21 |
| 1951 N Burling St | RM-4.5 | 99.7% | 143 yrs | Yes | 34 |
| 2624 N Racine Ave | RT-4 | 99.6% | 135 yrs | Yes | 7 |
| 1761 N Wells St | RM-5 | 99.5% | 132 yrs | Yes | 4 |
| 1827 N Clybourn Ave | B1-2 | 99.4% | 135 yrs | Yes | 24 |

### Building age is the strongest predictor

Building age supplanted land ratio as the single most predictive feature. The median building age among the top 500 parcels is 130 years — most were built in the late 1800s or early 1900s. At that age, deferred maintenance, rising land values, and ownership churn all converge. Exterior wall type also matters: masonry and frame construction appear independently in the SHAP rankings, with masonry being a stronger signal. Older masonry and frame structures are the ones most likely to be torn down.

### Land ratio still matters, but differently

Land ratio (land value ÷ total assessed value) dropped from #1 to #3 in importance after the removal of condos and the addition of new features. It remains a strong signal: the typical top-500 parcel has a land ratio of 0.51 — land accounts for more than half of total assessed value — compared to the citywide residential median of roughly 0.22. That 2.3x gap reflects the core teardown dynamic: the building contributes little to value, and the land itself is what developers are paying for.

| | All residential parcels | Top 500 high-risk parcels |
|--|------------------------|--------------------------|
| 25th percentile | ~0.12 | ~0.47 |
| **Median** | **~0.22** | **0.51** |
| 75th percentile | ~0.33 | ~0.60 |

### Two distinct demolition profiles in the top 500

**North Side teardowns** dominate the top of the list: Lake View (84), North Center (78), Lincoln Park (62), West Town (34), Logan Square (24). These are LLC-owned buildings — 50% of the top 500 have corporate ownership — on land that has appreciated beyond what the structure justifies. The common pattern is a 100–140-year-old two-to-six flat or small commercial building in an RS-3 or RT-4 zone, surrounded by active demolition and new construction activity.

**South Side distressed properties** make up a different cluster lower in the list: West Englewood (42) and Englewood (38) together account for 80 parcels. These aren't teardown-for-new-construction signals. They're properties with building violations, tax sale flags, and vacancy status — distress-driven demolitions rather than development-driven ones. The economic logic is different from the North Side: here, demolition often follows abandonment rather than preceding redevelopment.

**Property type breakdown:** Class 2-11 (apartment buildings with 2–6 units) accounts for 273 of the 500 highest-risk parcels (55%). Class 2-02 (single-family homes under 1,000 sq ft) and 2-03 (one-story residences, 1,000–1,800 sq ft) account for 80 each. Of the full top 500: 167 parcels (33%) appeared in the 2024 annual tax sale as forfeited, and 23 are on the vacant building registry.

### Validation confirms the model finds real demolitions

The validation sample's known demolitions (30 parcels confirmed demolished in 2024–2026) are concentrated in North Center (7), Lake View (3), Logan Square (3), and Englewood (3) — consistent with where the model focuses attention. The median score for a confirmed demolished parcel in the validation set is 0.99, more than 20x the median for non-demolished parcels (0.05).

### The false positives have shifted from Roseland to the North Side

In prior runs, the highest-scoring parcels that were *not* demolished were concentrated in Roseland — parcels with many nearby demolitions and LLC ownership but without the development demand to trigger new construction. That pattern has changed.

The current top 10 false positives are now all on the North Side: North Center (3), West Town (2), Logan Square (2), Lincoln Park (2), Lake View (1). These are LLC-owned buildings, 119–143 years old, with 6–34 nearby demolitions — the same profile as the confirmed demolitions. They may simply not have received a permit yet rather than being genuine model errors. The model reads the same signals correctly; the timing is uncertain.

### Precision improved substantially

After excluding condos and rescoping to the 480,600 relevant parcels, the confusion matrix at p ≥ 0.15 shows 126,947 false positives — down 57% from 291,776 in the prior run, even though the outcome window now extends through 2026. The model continues to miss zero confirmed demolitions at this threshold.

## Background

This is an upgrade from an earlier logistic regression model (`demolition_model.py`). XGBoost was chosen for improved performance on this rare-event classification task. The parcel scope was refined to exclude condominiums (Cook County property class 2-99), which are assessed and demolished differently from residential buildings and vacant land, reducing noise in the training data. The outcome window was extended from 2024–2025 to 2024–2026 to capture demolitions that occur further out from the snapshot date.

The model was built for [Chicago Cityscape](https://www.chicagocityscape.com) to help identify properties at elevated demolition risk.
