"""
Supplement — Pooled Horizon Error Growth Shape

horizon_error_curve.py's per-location curves are noisy (only 3 anchor
years each) -- Pune's raw data even shows error DECREASING with horizon,
which is almost certainly noise, not a real effect (there's no physical
reason far-future forecasts would be more reliable than near-future ones).

This pools relative growth (each location/anchor's error normalized by
its own year-1 error) across all 3 locations x 3 anchors = 9 curves,
giving a much more reliable estimate of the GROWTH SHAPE. It then
enforces monotonic non-decrease (cumulative max) -- an explicit modeling
choice reflecting the physical expectation that uncertainty should not
shrink with horizon, since the raw pooled data alone doesn't cleanly
guarantee this given the small sample.

The pooled shape is then applied to each location's OWN baseline error
level (so Kolkata still ends up with wider bands than Jaisalmer/Pune --
we're only borrowing the growth SHAPE across sites, not the absolute
error level).

Run:
    python pooled_horizon_curve.py
"""

import numpy as np
import pandas as pd

detail = pd.read_csv("data/horizon_error_detail.csv")

# Normalize each (location, anchor) curve by its own horizon_year=1 value
normalized_rows = []
for (location_id, anchor), group in detail.groupby(["location_id", "anchor_year"]):
    group = group.sort_values("horizon_year")
    year1_nrmse = group.iloc[0]["nrmse"]
    for _, row in group.iterrows():
        normalized_rows.append({
            "location_id": location_id, "anchor_year": anchor,
            "horizon_year": row["horizon_year"],
            "relative_error": row["nrmse"] / year1_nrmse,
        })

normalized_df = pd.DataFrame(normalized_rows)

# Pool across ALL locations and anchors -> one shared growth shape
pooled_shape = normalized_df.groupby("horizon_year")["relative_error"].mean().reset_index()
pooled_shape = pooled_shape.sort_values("horizon_year").reset_index(drop=True)

# Enforce monotonic non-decrease -- explicit modeling choice (see docstring)
pooled_shape["relative_error_monotonic"] = pooled_shape["relative_error"].cummax()

print("=== Pooled relative error growth shape (across all 3 locations x 3 anchors) ===")
print(pooled_shape.to_string(index=False))

# Each location's baseline (year-1) error level, to scale the shape by
baseline_by_location = detail[detail["horizon_year"] == 1].groupby("location_id")["nrmse"].mean()

print("\n=== Each location's year-1 baseline nRMSE ===")
print(baseline_by_location.to_string())

# Build the final per-location, per-horizon-year error curve that
# monte_carlo_forecast.py will actually use
final_rows = []
for location_id, baseline in baseline_by_location.items():
    for _, row in pooled_shape.iterrows():
        final_rows.append({
            "location_id": location_id,
            "horizon_year": int(row["horizon_year"]),
            "nrmse": baseline * row["relative_error_monotonic"],
        })

final_curve = pd.DataFrame(final_rows)
final_curve.to_csv("data/horizon_error_curve_final.csv", index=False)

print("\n=== Final per-location horizon error curve (shared shape x own baseline) ===")
for location_id, group in final_curve.groupby("location_id"):
    group = group.sort_values("horizon_year")
    print(f"\n{location_id}:")
    print(group[["horizon_year", "nrmse"]].to_string(index=False))

print("\nSaved to data/horizon_error_curve_final.csv")