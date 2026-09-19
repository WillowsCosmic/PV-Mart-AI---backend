"""
Supplement to Milestone 7/9 — Empirical Horizon Error Curve

Milestone 7's long-horizon simulation (Section 9.2) computed one POOLED
nRMSE across the full 10-year forecast window per anchor year. That
hides whether year-1 forecasts are more accurate than year-10 forecasts
-- which we need to know to build an honest Monte Carlo uncertainty band
(uncertainty should grow with horizon, not stay flat).

This script re-runs the same anchor-year simulations (2010, 2012, 2014),
but breaks the error down by INDIVIDUAL YEAR-AHEAD (horizon_year 1..10)
instead of pooling across the whole window, then averages each
horizon_year's error across the 3 anchors -- giving a real, data-driven
error-vs-horizon curve per location.

Run:
    python horizon_error_curve.py
"""

import warnings
import numpy as np
import pandas as pd

from site_configs import PILOT_SITES
from model_pool import add_lag_features, compute_metrics, CANDIDATES, TARGET_COLUMN, load_monthly

warnings.filterwarnings("ignore")

LONG_HORIZON_ANCHORS = [2010, 2012, 2014]
FORECAST_YEARS = 10


def compute_per_horizon_year_errors(site: dict) -> pd.DataFrame:
    location_id = site["location_id"]
    df = load_monthly(location_id)
    df = add_lag_features(df)

    rows = []
    for anchor in LONG_HORIZON_ANCHORS:
        train = df[df["year"] <= anchor]
        test = df[(df["year"] > anchor) & (df["year"] <= anchor + FORECAST_YEARS)]
        if len(test) == 0 or len(train) < 24:
            continue

        try:
            y_pred_monthly = CANDIDATES["SARIMA"](train, test)
        except Exception as exc:
            print(f"  {location_id} anchor {anchor}: SARIMA failed: {exc}")
            continue

        test = test.copy()
        test["y_pred"] = y_pred_monthly
        # horizon_year: 1 = the first forecasted year after the anchor, ... 10 = tenth
        test["horizon_year"] = test["year"] - anchor

        for h in range(1, FORECAST_YEARS + 1):
            year_slice = test[test["horizon_year"] == h]
            if year_slice.empty:
                continue
            y_true = year_slice[TARGET_COLUMN].values
            y_pred = year_slice["y_pred"].values
            metrics = compute_metrics(y_true, y_pred)
            rows.append({
                "location_id": location_id, "anchor_year": anchor,
                "horizon_year": h, "nrmse": metrics["nrmse"], "smape": metrics["smape"],
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_rows = []
    for site in PILOT_SITES:
        print(f"\n=== {site['location_id']} ({site['city']}) — per-horizon-year error ===")
        df = compute_per_horizon_year_errors(site)
        all_rows.append(df)

        # Average across the 3 anchors, per horizon year
        avg_by_horizon = df.groupby("horizon_year")[["nrmse", "smape"]].mean().reset_index()
        print(avg_by_horizon.to_string(index=False))

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv("data/horizon_error_detail.csv", index=False)

    # This is the table monte_carlo_forecast.py will actually consume:
    # one averaged nRMSE curve per location, indexed by horizon_year 1-10
    curve = full_df.groupby(["location_id", "horizon_year"])["nrmse"].mean().reset_index()
    curve.to_csv("data/horizon_error_curve.csv", index=False)

    print("\n=== Does error actually grow with horizon? (year-1 vs year-10 nRMSE) ===")
    for location_id, group in curve.groupby("location_id"):
        group = group.sort_values("horizon_year")
        y1 = group.iloc[0]["nrmse"]
        y10 = group[group["horizon_year"] == 10]
        y10_val = y10["nrmse"].values[0] if not y10.empty else group.iloc[-1]["nrmse"]
        print(f"  {location_id}: year 1 nRMSE={y1:.4f}  ->  year {group.iloc[-1]['horizon_year']:.0f} nRMSE={y10_val:.4f}")

    print("\nEmpirical horizon error curve saved to data/horizon_error_curve.csv")