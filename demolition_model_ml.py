#!/usr/bin/env python3
"""
Chicago Demolition Probability Model — XGBoost Edition

Gradient boosting model that estimates the probability any given Chicago parcel
will be demolished within a 3-year window. Trains on features as of end-2023,
validates against actual demolitions in 2024-2025.

Upgraded from logistic regression (demolition_model.py) to XGBoost for
improved pure-prediction performance on this rare-event classification task.

Usage:
    python3 cli/demolition_model_ml.py

Requires environment variables (or .env file):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Or pass connection string directly:
    python3 cli/demolition_model_ml.py --dsn "host=... port=... dbname=... user=... password=..."

Dependencies:
    pip install psycopg2-binary pandas numpy scikit-learn xgboost shap

Output files (written to cli/output/):
    - demolition_model_ml_top500.csv         Top 500 highest-risk active parcels
    - demolition_model_ml_importance.csv     SHAP feature importances
    - demolition_model_ml_validation.csv     Known demolitions + false positives with scores
    - demolition_model_ml_metrics.txt        All evaluation metrics
"""

import argparse
import csv
from datetime import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import shap
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_csv(df, path):
    """Write a DataFrame to CSV using Python's built-in csv module.

    Avoids pandas .to_csv(), which triggers an ImportError on older Anaconda
    pandas builds (missing SequenceNotStr in pandas._typing).
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(df.columns.tolist())
        for row in df.itertuples(index=False, name=None):
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# NOTE: SNAPSHOT_YEAR is set to 2023 because that is the earliest year available
# in both assessor_assessed_values2 (2021+) and assessor_single_mf_characteristics
# (2023+). Historical 2019 data could be fetched and loaded to restore the original
# 2019→2020-2022 training/validation window, but that import has not been done yet.
SNAPSHOT_YEAR = 2023
OUTCOME_START = "2024-01-01"
OUTCOME_END = "2025-12-31"
RANDOM_SEED = 42

# Zoning table name (current as of March 2026)
ZONING_TABLE = "zoning_20260218_010859"
ZONING_LOOKUP_TABLE = "zoning_lookup_20220829"

# Output directory (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Numeric features used in the model
NUMERIC_FEATURES = [
    "land_ratio",
    "land_val",
    "building_val",
    "building_age",
    "underbuilt_ratio",
    "violation_count_5yr",
    "has_open_violation",
    "years_since_renovation",
    "renovation_investment",
    "nearby_demo_count_2yr",
    "nearby_new_construction_count",
    "is_llc_owner",
    "is_vacant",
    "in_tax_sale",
    "sale_year",
    "sale_price",
    "sale_price_to_assessed_ratio",
    "lot_size_sf",
]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_connection(dsn=None):
    """Create a database connection.

    Priority order:
    1. Explicit --dsn argument
    2. ~/.pgpass: looks for the cityscape user on the production DigitalOcean host
    3. Environment variables DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
    """
    if dsn:
        return psycopg2.connect(dsn)

    # Try to read credentials from ~/.pgpass.
    # .pgpass format: hostname:port:database:username:password
    # We look for the line matching the production host and the cityscape user.
    PGPASS_HOST = "db-postgresql-nyc3-50995-do-user-666342-0.b.db.ondigitalocean.com"
    PGPASS_USER = "cityscape"
    PGPASS_DB = "defaultdb"
    PGPASS_PORT = "25060"

    pgpass_path = Path.home() / ".pgpass"
    if pgpass_path.exists():
        with open(pgpass_path) as f:
            for line in f:
                line = line.strip()
                # Skip blank lines and comments
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                if len(parts) != 5:
                    continue
                h, port, db, user, password = parts
                # Match on host and username; accept wildcard (*) for db
                if h == PGPASS_HOST and user == PGPASS_USER:
                    resolved_db = PGPASS_DB if db == "*" else db
                    conn_str = (
                        f"host={h} port={port} dbname={resolved_db} user={user} password={password} "
                        f"connect_timeout=30 options='-c statement_timeout=600000'"
                    )
                    return psycopg2.connect(conn_str)
        # If we reach here, no matching line was found in .pgpass
        raise RuntimeError(
            f"No matching .pgpass entry for host={PGPASS_HOST} user={PGPASS_USER}. "
            "Add it or pass --dsn."
        )

    # Fall back to environment variables
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "25060")
    dbname = os.getenv("DB_NAME", "defaultdb")
    user = os.getenv("DB_USER", "cityscape")
    password = os.getenv("DB_PASSWORD", "")

    if not host:
        raise RuntimeError(
            "No database connection info. Add an entry to ~/.pgpass, set "
            "DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD environment variables, "
            "or pass --dsn."
        )

    conn_str = (
        f"host={host} port={port} dbname={dbname} user={user} password={password} "
        f"connect_timeout=30 options='-c statement_timeout=600000'"
    )
    return psycopg2.connect(conn_str)


def run_query(conn, sql, description="query"):
    """Execute a SQL query and return a pandas DataFrame.

    Uses a psycopg2 cursor directly to avoid the SQLAlchemy deprecation
    warning that pd.read_sql_query() emits when passed a raw DBAPI2 connection.
    """
    print(f"  Running: {description} ... ", end="", flush=True)
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    elapsed = time.time() - t0
    print(f"{len(df):,} rows in {elapsed:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Step 1: Extract features from the database
# (Identical to demolition_model.py — only the model step changes)
# ---------------------------------------------------------------------------


def extract_features(conn):
    """Run all feature-extraction queries and return a merged DataFrame."""

    # ------------------------------------------------------------------
    # 1a. Base parcels: all Chicago parcels from propertytaxes_combined
    # ------------------------------------------------------------------
    print("\n[1/11] Base parcels")
    df_base = run_query(
        conn,
        """
        SELECT
            pin14,
            pin10,
            address,
            city,
            building_age AS building_age_raw,
            building_size AS building_size_raw,
            lot_size AS lot_size_raw,
            property_class,
            address_latest->>'name' AS taxpayer_name
        FROM propertytaxes_combined
        WHERE city = 'CHICAGO'
          AND deleted_at IS NULL
          AND geom_2025 IS NOT NULL
        """,
        "base parcels",
    )

    # Parse is_llc_owner from taxpayer name
    df_base["is_llc_owner"] = (
        df_base["taxpayer_name"]
        .fillna("")
        .str.upper()
        .str.contains(r"LLC|INC|CORP|TRUST", regex=True)
        .astype(int)
    )

    # ------------------------------------------------------------------
    # 1b. Assessed values for land_ratio (snapshot year)
    # ------------------------------------------------------------------
    print("\n[2/11] Assessed values (land_ratio)")
    df_av = run_query(
        conn,
        f"""
        SELECT DISTINCT ON (pin)
            pin AS pin14,
            COALESCE(board_land, certified_land, mailed_land) AS land_val,
            COALESCE(board_tot, certified_tot, mailed_tot) AS total_val
        FROM assessor_assessed_values2
        WHERE year <= {SNAPSHOT_YEAR}
        ORDER BY pin, year DESC
        """,
        "assessed values",
    )
    # Compute land_ratio: land assessed value / total assessed value
    df_av["land_ratio"] = np.where(
        df_av["total_val"].fillna(0) > 0,
        df_av["land_val"].astype(float) / df_av["total_val"].astype(float),
        np.nan,
    )
    df_av["land_val"] = pd.to_numeric(df_av["land_val"], errors="coerce")
    df_av["total_val"] = pd.to_numeric(df_av["total_val"], errors="coerce")
    df_av["building_val"] = df_av["total_val"] - df_av["land_val"]
    df_av = df_av[["pin14", "land_ratio", "land_val", "building_val", "total_val"]]

    # ------------------------------------------------------------------
    # 1c. Building characteristics (year built, building sqft, land sqft)
    # ------------------------------------------------------------------
    print("\n[3/11] Building characteristics")
    df_chars = run_query(
        conn,
        f"""
        SELECT DISTINCT ON (pin)
            pin AS pin14,
            char_yrblt,
            char_bldg_sf,
            char_land_sf
        FROM assessor_single_mf_characteristics
        WHERE year <= {SNAPSHOT_YEAR}
        ORDER BY pin, year DESC
        """,
        "building characteristics",
    )
    # Compute building_age
    df_chars["building_age"] = np.where(
        pd.to_numeric(df_chars["char_yrblt"], errors="coerce") > 0,
        SNAPSHOT_YEAR - pd.to_numeric(df_chars["char_yrblt"], errors="coerce"),
        np.nan,
    )

    # ------------------------------------------------------------------
    # 1d. Demolition permits (outcome variable)
    # ------------------------------------------------------------------
    # Three matching strategies in priority order:
    #   1. Address match (case insensitive) — most reliable; catches permits
    #      issued after 2023-10-12 when pins_jsonb population was disabled
    #   2. pins_jsonb match — reliable for permits issued before 2023-10-12
    #   3. Spatial match (ST_Intersects against parcel_2025) — fallback only
    #      for permits unmatched by address or pins_jsonb; the permit point
    #      geometry can fall on a neighboring parcel, so address is preferred
    #
    # NOT EXISTS on the spatial branch avoids matching permits already
    # resolved by address or pins_jsonb.
    print(f"\n[4/11] Demolition permits (outcome) {OUTCOME_START} to {OUTCOME_END}")
    df_demo = run_query(
        conn,
        f"""
        WITH

        matched_by_1_or_2 AS (
            SELECT DISTINCT permit_
            FROM (
                -- Strategy 1: address match
                SELECT p.permit_
                FROM permits p
                JOIN propertytaxes_combined pt
                    ON LOWER(p.address) = LOWER(pt.address)
                WHERE p._permit_type = 'PERMIT - WRECKING/DEMOLITION'
                  AND p.issue_date BETWEEN '{OUTCOME_START}' AND '{OUTCOME_END}'
                  AND pt.city = 'CHICAGO'
                  AND pt.deleted_at IS NULL

                UNION ALL

                -- Strategy 2: pins_jsonb match
                SELECT permit_
                FROM permits
                WHERE _permit_type = 'PERMIT - WRECKING/DEMOLITION'
                  AND issue_date BETWEEN '{OUTCOME_START}' AND '{OUTCOME_END}'
                  AND pins_jsonb IS NOT NULL
            ) m
        )

        SELECT DISTINCT pin14, 1 AS demolished_within_3yr
        FROM (
            -- Strategy 1: address match
            SELECT pt.pin14
            FROM permits p
            JOIN propertytaxes_combined pt
                ON LOWER(p.address) = LOWER(pt.address)
            WHERE p._permit_type = 'PERMIT - WRECKING/DEMOLITION'
              AND p.issue_date BETWEEN '{OUTCOME_START}' AND '{OUTCOME_END}'
              AND pt.city = 'CHICAGO'
              AND pt.deleted_at IS NULL

            UNION ALL

            -- Strategy 2: pins_jsonb match
            SELECT jsonb_array_elements_text(pins_jsonb) AS pin14
            FROM permits
            WHERE _permit_type = 'PERMIT - WRECKING/DEMOLITION'
              AND issue_date BETWEEN '{OUTCOME_START}' AND '{OUTCOME_END}'
              AND pins_jsonb IS NOT NULL

            UNION ALL

            -- Strategy 3: spatial match, only for permits not matched above
            SELECT p.pin14
            FROM permits pm
            JOIN parcel_2025 p ON ST_Intersects(p.geom, pm.geom_3435)
            WHERE pm._permit_type = 'PERMIT - WRECKING/DEMOLITION'
              AND pm.issue_date BETWEEN '{OUTCOME_START}' AND '{OUTCOME_END}'
              AND pm.geom_3435 IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM matched_by_1_or_2 m WHERE m.permit_ = pm.permit_
              )
        ) sub
        """,
        f"demolition permits {OUTCOME_START[:4]}-{OUTCOME_END[:4]}",
    )
    # Deduplicate in case a PIN appears in multiple demo permits
    df_demo = df_demo.drop_duplicates(subset=["pin14"])

    # ------------------------------------------------------------------
    # 1e. Violation counts (5-year window before snapshot)
    # ------------------------------------------------------------------
    # Join violations to parcels by address rather than spatially. Many violations
    # are geocoded to a point on the street centerline which falls outside the
    # parcel boundary, so a spatial intersection misses them. Both tables store
    # addresses in the same uppercase format, and violations has an index on
    # lower(address), so this join is efficient.
    print("\n[5/11] Violation counts (address join)")
    df_violations = run_query(
        conn,
        f"""
        SELECT
            pt.pin14,
            COUNT(*) AS violation_count_5yr,
            SUM(CASE WHEN v.violation_status = 'OPEN' THEN 1 ELSE 0 END) AS open_violation_count
        FROM violations v
        INNER JOIN propertytaxes_combined pt
            ON LOWER(pt.address) = LOWER(v.address)
        WHERE v.violation_date BETWEEN '{SNAPSHOT_YEAR - 5}-01-01' AND '{SNAPSHOT_YEAR}-12-31'
          AND pt.city = 'CHICAGO'
          AND pt.deleted_at IS NULL
        GROUP BY pt.pin14
        """,
        "violations (address join)",
    )
    df_violations["has_open_violation"] = (
        df_violations["open_violation_count"] > 0
    ).astype(int)

    # ------------------------------------------------------------------
    # 1f. Renovation permits (years since renovation + investment)
    # ------------------------------------------------------------------
    # Three matching strategies, in priority order:
    #   1. Address match (case insensitive) — most reliable for older permits
    #   2. pins_jsonb match — reliable for permits issued before 2023-10-12
    #   3. Spatial match within 10 feet of the parcel polygon — only applied
    #      to permits that had no match from strategies 1 or 2, avoiding the
    #      expensive spatial join on the full permit set.
    # UNION ALL + DISTINCT ON (pin14, permit_) ensures each permit is
    # counted once per parcel regardless of how many methods matched it.
    print("\n[6/11] Renovation permits (address + pins_jsonb + spatial fallback)")
    df_reno = run_query(
        conn,
        f"""
        WITH

        -- Collect every permit_ that matched via address or pins_jsonb.
        -- Used to exclude already-matched permits from the spatial fallback.
        matched_by_1_or_2 AS (
            SELECT DISTINCT permit_
            FROM (
                -- Strategy 1: case-insensitive address match
                SELECT p.permit_
                FROM permits p
                JOIN propertytaxes_combined pt
                    ON LOWER(p.address) = LOWER(pt.address)
                WHERE p._permit_type != 'PERMIT - WRECKING/DEMOLITION'
                  AND p.issue_date <= '{SNAPSHOT_YEAR}-12-31'
                  AND pt.city = 'CHICAGO'
                  AND pt.deleted_at IS NULL

                UNION ALL

                -- Strategy 2: pins_jsonb match
                SELECT permit_
                FROM permits
                WHERE _permit_type != 'PERMIT - WRECKING/DEMOLITION'
                  AND issue_date <= '{SNAPSHOT_YEAR}-12-31'
                  AND pins_jsonb IS NOT NULL
            ) m
        ),

        -- Strategy 1: case-insensitive address match
        addr_matches AS (
            SELECT
                pt.pin14,
                p.permit_,
                p.issue_date,
                p._estimated_cost
            FROM permits p
            JOIN propertytaxes_combined pt
                ON LOWER(p.address) = LOWER(pt.address)
            WHERE p._permit_type != 'PERMIT - WRECKING/DEMOLITION'
              AND p.issue_date <= '{SNAPSHOT_YEAR}-12-31'
              AND pt.city = 'CHICAGO'
              AND pt.deleted_at IS NULL
        ),

        -- Strategy 2: pins_jsonb match
        pins_matches AS (
            SELECT
                jsonb_array_elements_text(p.pins_jsonb) AS pin14,
                p.permit_,
                p.issue_date,
                p._estimated_cost
            FROM permits p
            WHERE p._permit_type != 'PERMIT - WRECKING/DEMOLITION'
              AND p.issue_date <= '{SNAPSHOT_YEAR}-12-31'
              AND p.pins_jsonb IS NOT NULL
        ),

        -- Strategy 3: spatial match, only for permits not already matched
        -- by address or pins_jsonb. NOT EXISTS filters those out before
        -- the expensive spatial join runs.
        spatial_matches AS (
            SELECT
                pt.pin14,
                p.permit_,
                p.issue_date,
                p._estimated_cost
            FROM permits p
            JOIN propertytaxes_combined pt
                ON ST_DWithin(p.geom_3435, pt.geom_2025, 10)
            WHERE p._permit_type != 'PERMIT - WRECKING/DEMOLITION'
              AND p.issue_date <= '{SNAPSHOT_YEAR}-12-31'
              AND p.geom_3435 IS NOT NULL
              AND pt.city = 'CHICAGO'
              AND pt.deleted_at IS NULL
              AND pt.geom_2025 IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM matched_by_1_or_2 m WHERE m.permit_ = p.permit_
              )
        )

        SELECT
            pin14,
            MAX(issue_date) AS last_renovation_date,
            SUM(CASE WHEN issue_date >= '{SNAPSHOT_YEAR - 5}-01-01'
                     THEN COALESCE(_estimated_cost, 0) ELSE 0 END) AS renovation_investment
        FROM (
            SELECT DISTINCT pin14, permit_, issue_date, _estimated_cost
            FROM (
                SELECT * FROM addr_matches
                UNION ALL
                SELECT * FROM pins_matches
                UNION ALL
                SELECT * FROM spatial_matches
            ) combined
        ) deduped
        GROUP BY pin14
        """,
        "renovation permits (address + pins_jsonb + spatial fallback)",
    )
    # Compute years_since_renovation
    df_reno["last_renovation_date"] = pd.to_datetime(
        df_reno["last_renovation_date"], errors="coerce"
    )
    snapshot_date = pd.Timestamp(f"{SNAPSHOT_YEAR}-12-31")
    df_reno["years_since_renovation"] = np.where(
        df_reno["last_renovation_date"].notna(),
        ((snapshot_date - df_reno["last_renovation_date"]).dt.days / 365.25).round(1),
        99.0,
    )
    df_reno = df_reno[["pin14", "years_since_renovation", "renovation_investment"]]

    # ------------------------------------------------------------------
    # 1g. Most recent arm's-length sale year
    # ------------------------------------------------------------------
    print("\n[7/11] Recent sales")
    df_sales = run_query(
        conn,
        f"""
        SELECT DISTINCT ON (pin14)
            pin14,
            EXTRACT(YEAR FROM daterecorded)::int AS sale_year,
            price_fullconsideration AS sale_price
        FROM view_ptax_cook_lake_idor
        WHERE daterecorded <= '{SNAPSHOT_YEAR}-12-31'
          AND price_fullconsideration > 10000
          AND line5_instrumenttype NOT IN ('Quit Claim Deed', 'Executor Deed', 'Beneficial interest')
          AND line5_instrumenttype IS NOT NULL
          AND sale_filter_ptax_flag IS FALSE
        ORDER BY pin14, daterecorded DESC
        """,
        "recent sales",
    )

    # ------------------------------------------------------------------
    # 1h. Vacant building registry
    # ------------------------------------------------------------------
    print("\n[8/11] Vacant buildings")
    # d_chicago_vacantbuildingregistry_vbr_2026 is a current snapshot with no
    # historical date column, so we use all rows as a proxy for vacancy status.
    df_vacant = run_query(
        conn,
        """
        SELECT DISTINCT pin14, 1 AS is_vacant
        FROM d_chicago_vacantbuildingregistry_vbr_2026
        WHERE pin14 IS NOT NULL
          AND pin14 != ''
        """,
        "vacant buildings",
    )

    # ------------------------------------------------------------------
    # 1i. Zoning + max FAR (spatial join)
    # ------------------------------------------------------------------
    print("\n[9/11] Zoning + FAR (spatial)")
    df_zoning = run_query(
        conn,
        f"""
        SELECT DISTINCT ON (p.pin14)
            p.pin14,
            z.zone_class,
            zl.far AS max_far
        FROM propertytaxes_combined p
        JOIN {ZONING_TABLE} z
            ON ST_Intersects(ST_Centroid(p.geom_2025), z.geom)
        LEFT JOIN {ZONING_LOOKUP_TABLE} zl
            ON z.zone_class = zl.zone_class
        WHERE p.city = 'CHICAGO'
          AND p.deleted_at IS NULL
          AND p.geom_2025 IS NOT NULL
        ORDER BY p.pin14, zl.far DESC NULLS LAST
        """,
        "zoning + FAR",
    )

    # ------------------------------------------------------------------
    # 1j. Community area (spatial join)
    # ------------------------------------------------------------------
    print("\n[10/11] Community area (spatial)")
    df_community = run_query(
        conn,
        """
        SELECT DISTINCT ON (p.pin14)
            p.pin14,
            ca.community AS community_area
        FROM propertytaxes_combined p
        JOIN communityarea ca
            ON ST_Contains(ca.geom, ST_Centroid(p.geom_2025))
        WHERE p.city = 'CHICAGO'
          AND p.deleted_at IS NULL
          AND p.geom_2025 IS NOT NULL
        ORDER BY p.pin14
        """,
        "community area",
    )

    # ------------------------------------------------------------------
    # 1k. Nearby demolition count (spatial, expensive)
    # ------------------------------------------------------------------
    print("\n[11/11] Nearby demolitions (spatial, may be slow)")
    try:
        # Use permits.geom_3435 directly (GiST indexed) instead of joining
        # through propertytaxes_combined for the demo permit geometries.
        # 500 meters = ~1640 feet in SRID 3435 (Illinois State Plane, feet).
        df_nearby = run_query(
            conn,
            f"""
            WITH demo_permits AS (
                SELECT DISTINCT permit_, geom_3435
                FROM permits
                WHERE _permit_type = 'PERMIT - WRECKING/DEMOLITION'
                  AND issue_date BETWEEN '{SNAPSHOT_YEAR - 1}-01-01' AND '{SNAPSHOT_YEAR}-12-31'
                  AND geom_3435 IS NOT NULL
            )
            SELECT
                pt.pin14,
                COUNT(DISTINCT dp.permit_) AS nearby_demo_count_2yr
            FROM propertytaxes_combined pt
            JOIN demo_permits dp
                ON ST_DWithin(pt.geom_2025, dp.geom_3435, 1640)
            WHERE pt.city = 'CHICAGO'
              AND pt.deleted_at IS NULL
              AND pt.geom_2025 IS NOT NULL
            GROUP BY pt.pin14
            """,
            "nearby demolitions (500m radius)",
        )
    except Exception as e:
        # Fallback: count demolitions in same community area
        print(f"\n  WARNING: Spatial nearby query failed ({e}). Using community-area fallback.")
        df_nearby = run_query(
            conn,
            f"""
            WITH demo_pins AS (
                SELECT DISTINCT jsonb_array_elements_text(pins_jsonb) AS pin14
                FROM permits
                WHERE _permit_type = 'PERMIT - WRECKING/DEMOLITION'
                  AND issue_date BETWEEN '{SNAPSHOT_YEAR - 1}-01-01' AND '{SNAPSHOT_YEAR}-12-31'
                  AND pins_jsonb IS NOT NULL
            ),
            demo_with_ca AS (
                SELECT dp.pin14, ca.community AS community_area
                FROM demo_pins dp
                JOIN propertytaxes_combined p ON p.pin14 = dp.pin14
                JOIN communityarea ca ON ST_Contains(ca.geom, ST_Centroid(p.geom_2025))
                WHERE p.city = 'CHICAGO' AND p.deleted_at IS NULL
            )
            SELECT
                pc.pin14,
                COALESCE(dca.demo_count, 0) AS nearby_demo_count_2yr
            FROM (
                SELECT DISTINCT ON (p.pin14)
                    p.pin14, ca.community AS community_area
                FROM propertytaxes_combined p
                JOIN communityarea ca ON ST_Contains(ca.geom, ST_Centroid(p.geom_2025))
                WHERE p.city = 'CHICAGO' AND p.deleted_at IS NULL
                ORDER BY p.pin14
            ) pc
            LEFT JOIN (
                SELECT community_area, COUNT(*) AS demo_count
                FROM demo_with_ca
                GROUP BY community_area
            ) dca ON pc.community_area = dca.community_area
            WHERE dca.demo_count > 0
            """,
            "nearby demos (community area fallback)",
        )

    # ------------------------------------------------------------------
    # 1l. Nearby new construction count (spatial)
    # ------------------------------------------------------------------
    print("\n[12/12] Nearby new construction (spatial, may be slow)")
    try:
        df_nearby_new = run_query(
            conn,
            f"""
            WITH new_permits AS (
                SELECT DISTINCT permit_, geom_3435
                FROM permits
                WHERE _permit_type = 'PERMIT - NEW CONSTRUCTION'
                  AND issue_date BETWEEN '{SNAPSHOT_YEAR - 1}-01-01' AND '{SNAPSHOT_YEAR}-12-31'
                  AND geom_3435 IS NOT NULL
            )
            SELECT
                pt.pin14,
                COUNT(DISTINCT np.permit_) AS nearby_new_construction_count
            FROM propertytaxes_combined pt
            JOIN new_permits np
                ON ST_DWithin(pt.geom_2025, np.geom_3435, 1640)
            WHERE pt.city = 'CHICAGO'
              AND pt.deleted_at IS NULL
              AND pt.geom_2025 IS NOT NULL
            GROUP BY pt.pin14
            """,
            "nearby new construction (500m radius)",
        )
    except Exception as e:
        print(f"\n  WARNING: Spatial nearby new construction query failed ({e}). Skipping.")
        df_nearby_new = pd.DataFrame(columns=["pin14", "nearby_new_construction_count"])

    # ------------------------------------------------------------------
    # 1m. Tax sale presence (2024 annual tax sale)
    # ------------------------------------------------------------------
    print("\n[13/13] Tax sale")
    df_tax_sale = run_query(
        conn,
        """
        SELECT DISTINCT pin14, 1 AS in_tax_sale
        FROM d_annual_tax_sale_2024_results
        WHERE pin14 IS NOT NULL
          AND status = 'Forfeited'
        """,
        "tax sale",
    )

    # ------------------------------------------------------------------
    # Merge everything together
    # ------------------------------------------------------------------
    print("\nMerging all features ...")
    df = df_base.copy()

    # Left-join each feature set on pin14
    for feat_df in [
        df_av,
        df_chars[["pin14", "building_age", "char_bldg_sf", "char_land_sf"]],
        df_demo,
        df_violations[["pin14", "violation_count_5yr", "has_open_violation"]],
        df_reno,
        df_sales,
        df_vacant,
        df_zoning,
        df_community,
        df_nearby,
        df_nearby_new,
        df_tax_sale,
    ]:
        df = df.merge(feat_df, on="pin14", how="left")

    # Fill NaN for count/flag features with 0
    fill_zero_cols = [
        "demolished_within_3yr",
        "violation_count_5yr",
        "has_open_violation",
        "renovation_investment",
        "nearby_demo_count_2yr",
        "nearby_new_construction_count",
        "is_vacant",
        "in_tax_sale",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)

    # Fill years_since_renovation with 99 for parcels with no renovation permit
    if "years_since_renovation" in df.columns:
        df["years_since_renovation"] = df["years_since_renovation"].fillna(99.0)

    # lot_size_sf: direct pass-through of char_land_sf
    df["char_land_sf"] = pd.to_numeric(df.get("char_land_sf"), errors="coerce")
    df["lot_size_sf"] = df["char_land_sf"]

    # Compute underbuilt_ratio: actual building FAR / max zoning FAR
    df["char_bldg_sf"] = pd.to_numeric(df.get("char_bldg_sf"), errors="coerce")
    df["max_far"] = pd.to_numeric(df.get("max_far"), errors="coerce")

    df["underbuilt_ratio"] = np.where(
        (df["max_far"] > 0) & (df["char_land_sf"] > 0),
        (df["char_bldg_sf"] / df["char_land_sf"]) / df["max_far"],
        np.nan,
    )

    # sale_price_to_assessed_ratio: sale price relative to total assessed value.
    # Note: for class 2 (residential) properties, Cook County assesses at ~10% of
    # estimated market value, so this ratio will be ~10x higher than for classes
    # assessed closer to market. The model learns this relationship via property_class.
    df["sale_price"] = pd.to_numeric(df.get("sale_price"), errors="coerce")
    df["sale_price_to_assessed_ratio"] = np.where(
        df["total_val"].fillna(0) > 0,
        df["sale_price"] / df["total_val"],
        np.nan,
    )

    print(f"  Final dataset: {len(df):,} parcels, {df['demolished_within_3yr'].sum():.0f} demolished")
    return df


# ---------------------------------------------------------------------------
# Step 2-5: Model training, evaluation, and export
# ---------------------------------------------------------------------------


def train_and_evaluate(df, skip_shap=False, suffix=""):
    """Train the XGBoost model and evaluate on the full dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Prepare features and target
    # ------------------------------------------------------------------
    features = NUMERIC_FEATURES.copy()

    # Add community_area as one-hot encoded dummies (top 30 by frequency).
    # XGBoost can handle categoricals natively but one-hot encoding keeps
    # the pipeline consistent with the logistic regression baseline.
    if "community_area" in df.columns:
        top_communities = (
            df["community_area"].value_counts().head(30).index.tolist()
        )
        df["community_area_clean"] = df["community_area"].where(
            df["community_area"].isin(top_communities), other="OTHER"
        )
        community_dummies = pd.get_dummies(
            df["community_area_clean"], prefix="ca", dtype=float
        )
        df = pd.concat([df, community_dummies], axis=1)
        community_features = community_dummies.columns.tolist()
        features += community_features

    # Add property_class as one-hot encoded dummies (top 20 by frequency).
    if "property_class" in df.columns:
        top_classes = (
            df["property_class"].value_counts().head(20).index.tolist()
        )
        df["property_class_clean"] = df["property_class"].where(
            df["property_class"].isin(top_classes), other="OTHER"
        )
        class_dummies = pd.get_dummies(
            df["property_class_clean"], prefix="pc", dtype=float
        )
        df = pd.concat([df, class_dummies], axis=1)
        features += class_dummies.columns.tolist()

    y = df["demolished_within_3yr"].values.astype(int)
    X = df[features].copy()

    # Impute missing numeric values with median before passing to XGBoost.
    # XGBoost can handle NaN natively via its own missing-value path, but
    # explicit imputation keeps behavior predictable and consistent with
    # the logistic regression baseline.
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), columns=features, index=X.index
    )

    # ------------------------------------------------------------------
    # Class imbalance: compute scale_pos_weight
    # ------------------------------------------------------------------
    # scale_pos_weight tells XGBoost how much to up-weight the minority class
    # (demolished parcels). This is the XGBoost equivalent of class_weight='balanced'
    # in sklearn. With ~1,000:1 imbalance, this is critical for recall.
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"\n  Class balance: {n_pos:,} demolished / {n_neg:,} not-demolished")
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    # ------------------------------------------------------------------
    # Train XGBoost model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST MODEL")
    print("=" * 70)

    model = xgb.XGBClassifier(
        n_estimators=500,         # number of boosting rounds
        max_depth=6,              # maximum tree depth; controls overfitting
        learning_rate=0.05,       # step shrinkage; lower = more conservative
        subsample=0.8,            # fraction of rows sampled per tree
        colsample_bytree=0.8,     # fraction of features sampled per tree
        scale_pos_weight=scale_pos_weight,  # handles severe class imbalance
        eval_metric="aucpr",      # optimize for PR-AUC, best metric for rare events
        random_state=RANDOM_SEED,
        n_jobs=-1,                # use all available CPU cores
        verbosity=0,              # suppress XGBoost's own progress output
    )
    model.fit(X_imputed, y)

    # ------------------------------------------------------------------
    # SHAP feature importance
    # ------------------------------------------------------------------
    # SHAP (SHapley Additive exPlanations) gives each feature a contribution
    # score per prediction. The mean absolute SHAP value across all parcels
    # is the best global importance metric — more informative than XGBoost's
    # built-in gain/cover because it accounts for feature interactions.
    importance_path = OUTPUT_DIR / f"demolition_model_ml_importance{suffix}.csv"
    if skip_shap and importance_path.exists():
        # Load previously computed importances instead of recomputing
        print(f"\nSkipping SHAP (--skip-shap set, loading {importance_path.name}) ...")
        importance_df = pd.read_csv(importance_path)
    else:
        print("\nComputing SHAP values (this may take a minute) ...")
        t0 = time.time()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_imputed)
        print(f"  SHAP done in {time.time() - t0:.1f}s")

        # Build importance table: mean |SHAP| per feature
        importance_df = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        write_csv(importance_df, importance_path)
        print(f"\n  Importances saved to {importance_path}")

    # Print only the core numeric features (not community area dummies)
    print("\n--- SHAP Feature Importance (core features, sorted) ---")
    core_importance = importance_df[importance_df["feature"].isin(NUMERIC_FEATURES)]
    for _, row in core_importance.iterrows():
        bar = "#" * int(row["mean_abs_shap"] * 200)  # visual bar
        print(f"  {row['feature']:30s}  {row['mean_abs_shap']:.6f}  {bar}")

    # ------------------------------------------------------------------
    # Predict probabilities on the full dataset
    # ------------------------------------------------------------------
    df["demolition_probability"] = model.predict_proba(X_imputed)[:, 1]

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)

    y_prob = df["demolition_probability"].values
    y_true = y

    # 1. ROC-AUC
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"\n  1. ROC-AUC:           {roc_auc:.4f}  (target > 0.80)")

    # 2. Average Precision (PR-AUC)
    pr_auc = average_precision_score(y_true, y_prob)
    base_rate = y_true.mean()
    print(f"  2. PR-AUC (Avg Prec): {pr_auc:.4f}  (base rate = {base_rate:.4f})")

    # 3. Top-decile lift: how much more likely are top-10% parcels to be demolished
    # compared to a random parcel? Target >10x for a good model.
    top_10_pct_threshold = np.percentile(y_prob, 90)
    top_10_mask = y_prob >= top_10_pct_threshold
    top_10_demo_rate = y_true[top_10_mask].mean()
    top_decile_lift = top_10_demo_rate / base_rate if base_rate > 0 else 0
    print(f"  3. Top-decile lift:   {top_decile_lift:.2f}x  (target > 10x)")
    print(f"     (demo rate in top 10%: {top_10_demo_rate:.4f})")

    # 4. Capture rate at top 5%: what fraction of all demolitions fall in the
    # top 5% of predicted scores? Target >0.40 for this model.
    top_5_pct_threshold = np.percentile(y_prob, 95)
    top_5_mask = y_prob >= top_5_pct_threshold
    total_demos = y_true.sum()
    captured_demos = y_true[top_5_mask].sum()
    capture_rate_5 = captured_demos / total_demos if total_demos > 0 else 0
    print(
        f"  4. Capture rate @5%:  {capture_rate_5:.4f} ({captured_demos:.0f}/{total_demos:.0f})  "
        f"(target > 0.40)"
    )

    # 5. Median predicted score separation between demolished and non-demolished parcels
    median_demo = np.median(y_prob[y_true == 1])
    median_nondemo = np.median(y_prob[y_true == 0])
    print(f"  5. Median score (demolished):     {median_demo:.4f}")
    print(f"     Median score (non-demolished): {median_nondemo:.4f}")
    print(f"     Separation ratio:              {median_demo / max(median_nondemo, 1e-9):.1f}x")

    # 6. Confusion matrix at p=0.15 threshold
    y_pred_15 = (y_prob >= 0.15).astype(int)
    cm = confusion_matrix(y_true, y_pred_15)
    print(f"\n  Confusion matrix at p >= 0.15 threshold:")
    print(f"                    Predicted NO   Predicted YES")
    print(f"    Actual NO       {cm[0, 0]:>10,}   {cm[0, 1]:>10,}")
    print(f"    Actual YES      {cm[1, 0]:>10,}   {cm[1, 1]:>10,}")

    # Save metrics to file
    metrics_path = OUTPUT_DIR / f"demolition_model_ml_metrics{suffix}.txt"
    with open(metrics_path, "w") as f:
        f.write("Chicago Demolition Probability Model (XGBoost) — Evaluation Metrics\n")
        f.write(f"Snapshot year: {SNAPSHOT_YEAR}\n")
        f.write(f"Outcome period: {OUTCOME_START} to {OUTCOME_END}\n")
        f.write(f"Total parcels: {len(df):,}\n")
        f.write(f"Total demolitions: {int(total_demos):,}\n")
        f.write(f"Base rate: {base_rate:.6f}\n")
        f.write(f"scale_pos_weight: {scale_pos_weight:.1f}\n\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC (Average Precision): {pr_auc:.4f}\n")
        f.write(f"Top-decile lift: {top_decile_lift:.2f}x\n")
        f.write(f"Capture rate at top 5%: {capture_rate_5:.4f} ({captured_demos:.0f}/{total_demos:.0f})\n")
        f.write(f"Median score (demolished): {median_demo:.4f}\n")
        f.write(f"Median score (non-demolished): {median_nondemo:.4f}\n\n")
        f.write(f"Confusion matrix at p >= 0.15:\n")
        f.write(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}\n")
        f.write(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}\n\n")
        f.write("SHAP feature importances (mean |SHAP|):\n")
        for _, row in importance_df[importance_df["feature"].isin(NUMERIC_FEATURES)].iterrows():
            f.write(f"  {row['feature']:30s}  {row['mean_abs_shap']:.6f}\n")
    print(f"\n  Metrics saved to {metrics_path}")

    return df, model, imputer


# ---------------------------------------------------------------------------
# Step 6: Score active parcels and export top 500
# ---------------------------------------------------------------------------


def export_top_500(df, suffix=""):
    """Export the top 500 highest-risk parcels that haven't been demolished."""

    print("\n" + "=" * 70)
    print("TOP 500 HIGHEST-RISK PARCELS")
    print("=" * 70)

    # Filter to parcels NOT demolished in the outcome period
    active = df[df["demolished_within_3yr"] == 0].copy()
    active = active.sort_values("demolition_probability", ascending=False)

    # Select columns for export
    export_cols = [
        "pin14",
        "pin10",
        "address",
        "demolition_probability",
        "land_ratio",
        "land_val",
        "building_val",
        "building_age",
        "underbuilt_ratio",
        "violation_count_5yr",
        "is_vacant",
        "in_tax_sale",
        "community_area",
        "zone_class",
        "property_class",
        "is_llc_owner",
        "nearby_demo_count_2yr",
        "nearby_new_construction_count",
        "years_since_renovation",
        "renovation_investment",
        "lot_size_sf",
        "sale_year",
        "sale_price",
        "sale_price_to_assessed_ratio",
    ]
    # Only include columns that exist
    export_cols = [c for c in export_cols if c in active.columns]

    top500 = active.head(500)[export_cols]
    top500_path = OUTPUT_DIR / f"demolition_model_ml_top500{suffix}.csv"
    write_csv(top500, top500_path)
    print(f"  Saved {len(top500)} parcels to {top500_path}")

    # Print summary of top 10
    print("\n  Top 10 highest-risk parcels:")
    print(f"  {'PIN14':20s} {'Prob':>8s} {'Land Ratio':>11s} {'Age':>5s} {'Violations':>11s} {'Community Area'}")
    for _, row in top500.head(10).iterrows():
        print(
            f"  {str(row.get('pin14','')):20s} "
            f"{row.get('demolition_probability', 0):8.4f} "
            f"{row.get('land_ratio', 0):11.4f} "
            f"{row.get('building_age', 0):5.0f} "
            f"{row.get('violation_count_5yr', 0):11.0f} "
            f"{str(row.get('community_area', '')):s}"
        )


# ---------------------------------------------------------------------------
# Step 7: Validation — known demolitions + false positives
# ---------------------------------------------------------------------------


def export_validation(df, suffix=""):
    """Export validation samples: known demolitions and high-scoring false positives."""

    print("\n" + "=" * 70)
    print("VALIDATION SAMPLES")
    print("=" * 70)

    val_cols = [
        "pin14",
        "pin10",
        "address",
        "demolished_within_3yr",
        "demolition_probability",
        "land_ratio",
        "land_val",
        "building_val",
        "building_age",
        "underbuilt_ratio",
        "violation_count_5yr",
        "has_open_violation",
        "years_since_renovation",
        "renovation_investment",
        "nearby_demo_count_2yr",
        "is_llc_owner",
        "is_vacant",
        "in_tax_sale",
        "sale_year",
        "sale_price",
        "sale_price_to_assessed_ratio",
        "lot_size_sf",
        "nearby_new_construction_count",
        "community_area",
        "zone_class",
        "property_class",
    ]
    val_cols = [c for c in val_cols if c in df.columns]

    # Known demolitions with their predicted scores
    demolished = df[df["demolished_within_3yr"] == 1].copy()
    demolished = demolished.sort_values("demolition_probability", ascending=False)

    # Sample 30: mix of high-scoring (correctly flagged) and low-scoring (missed)
    n_demo = len(demolished)
    if n_demo >= 30:
        # Take top 15, bottom 5, and 10 random from middle
        top_15 = demolished.head(15)
        bottom_5 = demolished.tail(5)
        middle = demolished.iloc[15:-5]
        if len(middle) >= 10:
            mid_10 = middle.sample(10, random_state=RANDOM_SEED)
        else:
            mid_10 = middle
        demo_sample = pd.concat([top_15, mid_10, bottom_5])
    else:
        demo_sample = demolished

    # 10 false positives: highest-scoring parcels that were NOT demolished
    not_demolished = df[df["demolished_within_3yr"] == 0].copy()
    not_demolished = not_demolished.sort_values("demolition_probability", ascending=False)
    fp_sample = not_demolished.head(10)

    # Combine and save
    validation = pd.concat(
        [
            demo_sample[val_cols].assign(sample_type="known_demolition"),
            fp_sample[val_cols].assign(sample_type="false_positive"),
        ]
    )
    val_path = OUTPUT_DIR / f"demolition_model_ml_validation{suffix}.csv"
    write_csv(validation, val_path)
    print(f"  Saved {len(validation)} validation samples to {val_path}")

    # Print known demolitions
    print(f"\n  Known demolitions ({len(demo_sample)} samples):")
    print(f"  {'PIN14':20s} {'Prob':>8s} {'Land Ratio':>11s} {'Age':>5s} {'Violations':>11s} {'Community Area'}")
    for _, row in demo_sample.head(20).iterrows():
        print(
            f"  {str(row.get('pin14','')):20s} "
            f"{row.get('demolition_probability', 0):8.4f} "
            f"{row.get('land_ratio', 0):11.4f} "
            f"{row.get('building_age', 0):5.0f} "
            f"{row.get('violation_count_5yr', 0):11.0f} "
            f"{str(row.get('community_area', '')):s}"
        )

    # Print false positives
    print(f"\n  False positives (top 10 high-scoring, not demolished):")
    print(f"  {'PIN14':20s} {'Prob':>8s} {'Land Ratio':>11s} {'Age':>5s} {'Violations':>11s} {'Community Area'}")
    for _, row in fp_sample.iterrows():
        print(
            f"  {str(row.get('pin14','')):20s} "
            f"{row.get('demolition_probability', 0):8.4f} "
            f"{row.get('land_ratio', 0):11.4f} "
            f"{row.get('building_age', 0):5.0f} "
            f"{row.get('violation_count_5yr', 0):11.0f} "
            f"{str(row.get('community_area', '')):s}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Chicago Demolition Probability Model (XGBoost)"
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=None,
        help="PostgreSQL connection string (alternative to env vars)",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip recomputing SHAP values and load the existing importance CSV instead",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append a timestamp suffix (e.g. _20260331_143022) to all output filenames",
    )
    args = parser.parse_args()

    suffix = datetime.now().strftime("_%Y%m%d_%H%M%S") if args.timestamp else ""

    print("=" * 70)
    print("CHICAGO DEMOLITION PROBABILITY MODEL (XGBoost)")
    print(f"Snapshot: end of {SNAPSHOT_YEAR}")
    print(f"Outcome window: {OUTCOME_START} to {OUTCOME_END}")
    print("=" * 70)

    # Connect to database
    print("\nConnecting to database ...")
    conn = get_connection(dsn=args.dsn)
    print("  Connected.")

    try:
        # Step 1: Extract features
        print("\n--- STEP 1: EXTRACTING FEATURES ---")
        df = extract_features(conn)

        # Steps 2-5: Train model and evaluate
        print("\n--- STEPS 2-5: TRAINING & EVALUATION ---")
        df, model, imputer = train_and_evaluate(df, skip_shap=args.skip_shap, suffix=suffix)

        # Step 6: Export top 500
        print("\n--- STEP 6: EXPORT TOP 500 ---")
        export_top_500(df, suffix=suffix)

        # Step 7: Validation
        print("\n--- STEP 7: VALIDATION ---")
        export_validation(df, suffix=suffix)

    finally:
        conn.close()
        print("\n  Database connection closed.")

    print("\n" + "=" * 70)
    print("DONE. Output files in:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
