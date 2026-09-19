"""
Milestone 7 / Stage 7 — Walk-Forward Validation

Replaces Milestone 6's single train/test split with genuine rolling-origin
(walk-forward) validation, per spec Section 9.1: the 26-year series is
NEVER randomly shuffled -- all splits are chronological, and the same
candidate pool is scored across MULTIPLE sliding windows, not just one.

Also implements Section 9.2's historical long-horizon simulation: pretend
it's an earlier year, train only on data up to that point, forecast
forward, compare to what actually happened -- repeated across every
feasible anchor year. This produces genuine out-of-sample, long-horizon
error statistics -- the only honest way to state a 10-year accuracy claim.

Run:
    python walk_forward.py
"""

import warnings
import numpy as np
import pandas as pd
import mlflow

from site_configs import PILOT_SITES
from model_pool import (
    add_lag_features, compute_metrics, CANDIDATES, FEATURE_COLUMNS,
    TARGET_COLUMN, MLFLOW_TRACKING_URI, load_monthly,
)

warnings.filterwarnings("ignore")

MLFLOW_EXPERIMENT = "solar-forecast-walk-forward"

# Rolling-origin folds (Section 9.1's example table, adapted to our
# 2001-2025 data window)
FOLDS = [
    {"train_end": 2016, "val_end": 2019, "test_end": 2022},
    {"train_end": 2019, "val_end": 2022, "test_end": 2025},
]

# Historical long-horizon simulation anchor years (Section 9.2): pretend
# it's this year, train up to it, forecast 10 years ahead, compare to
# what actually happened. Anchors need >=10 years of actual future data
# remaining in our 2001-2025 window to be checkable.
LONG_HORIZON_ANCHORS = [2010, 2012, 2014]  # each forecasts anchor+1..anchor+10


def run_walk_forward_folds(site: dict) -> pd.DataFrame:
    location_id = site["location_id"]
    df = load_monthly(location_id)
    df = add_lag_features(df)

    print(f"\n  {location_id} ({site['city']}) — walk-forward folds  ")

    rows = []
    for i, fold in enumerate(FOLDS, start=1):
        train = df[df["year"] <= fold["train_end"]]
        test = df[(df["year"] > fold["train_end"]) & (df["year"] <= fold["test_end"])]
        if len(test) == 0 or len(train) < 24:
            continue

        y_true = test[TARGET_COLUMN].values
        print(f"  Fold {i}: train <= {fold['train_end']}, "
              f"test {fold['train_end']+1}-{fold['test_end']}")

        for model_name, train_fn in CANDIDATES.items():
            try:
                y_pred = train_fn(train, test)
                metrics = compute_metrics(y_true, y_pred)
                rows.append({"location_id": location_id, "fold": i, "model": model_name, **metrics})
                print(f"    {model_name:16s} nRMSE={metrics['nrmse']:.4f}  sMAPE={metrics['smape']:.2f}%")
            except Exception as exc:
                print(f"    {model_name:16s} FAILED: {exc}")

    return pd.DataFrame(rows)


def run_long_horizon_simulation(site: dict) -> pd.DataFrame:
    """Section 9.2: pretend it's year X, train up to X, forecast 10 years
    forward, compare to actual. SARIMA (Milestone 6's most consistent
    winner) is used here -- this tests the REAL 10-year claim, not a
    3-year holdout."""
    location_id = site["location_id"]
    df = load_monthly(location_id)
    df = add_lag_features(df)

    print(f"\n  {location_id} — historical long-horizon simulation (Section 9.2)  ")

    rows = []
    for anchor in LONG_HORIZON_ANCHORS:
        train = df[df["year"] <= anchor]
        test = df[(df["year"] > anchor) & (df["year"] <= anchor + 10)]
        if len(test) == 0 or len(train) < 24:
            print(f"  Anchor {anchor}: insufficient data, skipping")
            continue

        y_true = test[TARGET_COLUMN].values
        try:
            y_pred = CANDIDATES["SARIMA"](train, test)
            metrics = compute_metrics(y_true, y_pred)
            rows.append({"location_id": location_id, "anchor_year": anchor,
                        "forecast_years": f"{anchor+1}-{anchor+10}", **metrics})
            print(f"  Anchor {anchor} -> forecast {anchor+1}-{anchor+10}: "
                  f"nRMSE={metrics['nrmse']:.4f}  sMAPE={metrics['smape']:.2f}%")
        except Exception as exc:
            print(f"  Anchor {anchor}: FAILED: {exc}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    all_fold_results = []
    all_long_horizon = []

    for site in PILOT_SITES:
        with mlflow.start_run(run_name=f"walkforward_{site['location_id']}"):
            fold_df = run_walk_forward_folds(site)
            all_fold_results.append(fold_df)
            mlflow.log_dict(fold_df.to_dict(), "fold_scoreboard.json")

        with mlflow.start_run(run_name=f"longhorizon_{site['location_id']}"):
            lh_df = run_long_horizon_simulation(site)
            all_long_horizon.append(lh_df)
            mlflow.log_dict(lh_df.to_dict(), "long_horizon_results.json")

    fold_scoreboard = pd.concat(all_fold_results, ignore_index=True)
    long_horizon_results = pd.concat(all_long_horizon, ignore_index=True)

    fold_scoreboard.to_csv("data/scoreboard_m7_folds.csv", index=False)
    long_horizon_results.to_csv("data/scoreboard_m7_longhorizon.csv", index=False)

    print("\n  Champion per location, averaged across ALL folds (not just one split)  ")
    avg_by_model = fold_scoreboard.groupby(["location_id", "model"])["nrmse"].mean().reset_index()
    for location_id, group in avg_by_model.groupby("location_id"):
        champion = group.loc[group["nrmse"].idxmin()]
        print(f"  {location_id}: {champion['model']} (avg nRMSE={champion['nrmse']:.4f} across folds)")

    print("\n Real 10-year-ahead error distribution (Section 9.2) ")
    print(long_horizon_results.groupby("location_id")[["nrmse", "smape"]].mean())

    print("\nMilestone 7 (walk-forward + long-horizon simulation) complete.")