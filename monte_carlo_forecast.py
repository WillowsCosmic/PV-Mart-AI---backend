"""
Milestone 9-10 / Stage 8 — 10-Year Probabilistic Forecast Output

Produces the final output: P50/P75/P90 per year for 2027-2036, generated
by Monte Carlo sampling over the three uncertainty sources named in
spec Section 10:

  (a) inter-annual resource variability, estimated from the 26-year
      historical distribution
  (b) model residual uncertainty, from the walk-forward validation error
      distribution (Milestone 7's real long-horizon error stats)
  (c) loss-parameter uncertainty (soiling range, degradation range)

Runs >=1,000 simulated 10-year trajectories per location and takes
percentiles.


"""

import json
import numpy as np
import pandas as pd

from site_configs import PILOT_SITES
from products_config import SITE_SYSTEM_CONFIG
from model_pool import load_monthly

MONTHLY_DIR = "data/monthly"
OUTPUT_DIR = "data"

N_SIMULATIONS = 1000
FORECAST_START_YEAR = 2027
FORECAST_YEARS = 10


SOILING_UNCERTAINTY = (-0.01, 0.01)        
DEGRADATION_RATE_RANGE = (0.004, 0.006)     

HORIZON_ERROR_CURVE = pd.read_csv("data/horizon_error_curve_final.csv")


def get_nrmse_for_horizon_year(location_id: str, horizon_year: int) -> float:
    row = HORIZON_ERROR_CURVE[
        (HORIZON_ERROR_CURVE["location_id"] == location_id) &
        (HORIZON_ERROR_CURVE["horizon_year"] == horizon_year)
    ]
    return row.iloc[0]["nrmse"]


def get_historical_annual_distribution(location_id: str) -> np.ndarray:
    """(a) Inter-annual resource variability: the actual year-to-year
    spread of annual generation across the full historical record."""
    df = load_monthly(location_id)
    annual = df.groupby("year")["PV_energy_kWh"].sum()
    return annual.values


def run_monte_carlo(location_id: str, system_kwp: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)

    historical_annual = get_historical_annual_distribution(location_id)
    baseline_mean = historical_annual.mean()
    baseline_std = historical_annual.std()

    trajectories = np.zeros((N_SIMULATIONS, FORECAST_YEARS))

    for sim in range(N_SIMULATIONS):
        degradation_rate = rng.uniform(*DEGRADATION_RATE_RANGE)
        soiling_offset = rng.uniform(*SOILING_UNCERTAINTY)

        for year_idx in range(FORECAST_YEARS):
            horizon_year = year_idx + 1  

            resource_draw = rng.normal(baseline_mean, baseline_std)

            nrmse_h = get_nrmse_for_horizon_year(location_id, horizon_year)
            model_error_std = baseline_mean * nrmse_h
            model_error = rng.normal(0, model_error_std)

            degradation_factor = (1 - degradation_rate) ** year_idx
            soiling_factor = 1 - soiling_offset

            annual_kwh = (resource_draw + model_error) * degradation_factor * soiling_factor
            trajectories[sim, year_idx] = max(0, annual_kwh)

    rows = []
    for year_idx in range(FORECAST_YEARS):
        year = FORECAST_START_YEAR + year_idx
        col = trajectories[:, year_idx]
        p50 = np.percentile(col, 50)
        p75 = np.percentile(col, 25)
        p90 = np.percentile(col, 10)
        p10 = np.percentile(col, 90)  # for CI width reporting

        rows.append({
            "year": year,
            "p50_kwh": round(p50, 1),
            "p75_kwh": round(p75, 1),
            "p90_kwh": round(p90, 1),
            "ci_width_kwh": round(p10 - p90, 1),
            "specific_yield_p50": round(p50 / system_kwp, 1),
            "cuf_pct_p50": round((p50 / (system_kwp * 8760)) * 100, 2),
        })

    return pd.DataFrame(rows)


def build_summary(forecast_df: pd.DataFrame, system_kwp: float) -> dict:
    total_10yr_kwh = forecast_df["p50_kwh"].sum()
    avg_annual = forecast_df["p50_kwh"].mean()

    first_year = forecast_df.iloc[0]["p50_kwh"]
    last_year = forecast_df.iloc[-1]["p50_kwh"]
    effective_degradation = (1 - (last_year / first_year) ** (1 / (FORECAST_YEARS - 1))) * 100

    return {
        "total_10yr_mwh": round(total_10yr_kwh / 1000, 2),
        "avg_annual_kwh": round(avg_annual, 1),
        "specific_yield_kwh_per_kwp": round(avg_annual / system_kwp, 1),
        "cuf_pct": round((avg_annual / (system_kwp * 8760)) * 100, 2),
        "avg_annual_degradation_pct": round(effective_degradation, 3),
        "ci_width_first_year_kwh": float(forecast_df.iloc[0]["ci_width_kwh"]),
        "ci_width_last_year_kwh": float(forecast_df.iloc[-1]["ci_width_kwh"]),
    }


if __name__ == "__main__":
    all_outputs = {}

    for site in PILOT_SITES:
        location_id = site["location_id"]
        system = SITE_SYSTEM_CONFIG[location_id]
        system_kwp = system["system_kwp"]

        print(f"\n=== {location_id} ({site['city']}) — "
              f"{N_SIMULATIONS} Monte Carlo trajectories, {system_kwp:.2f} kWp ===")

        forecast_df = run_monte_carlo(location_id, system_kwp)
        summary = build_summary(forecast_df, system_kwp)

        print(forecast_df[["year", "p50_kwh", "p75_kwh", "p90_kwh", "ci_width_kwh"]]
              .to_string(index=False))
        print(f"\n  Summary:")
        for k, v in summary.items():
            print(f"    {k}: {v}")

        forecast_df.to_csv(f"{OUTPUT_DIR}/forecast_{location_id}.csv", index=False)

        monthly_df = load_monthly(location_id)
        historical_monthly_avg = (
            monthly_df.groupby("month")["PV_energy_kWh"]
            .mean().round(1).reset_index()
            .rename(columns={"PV_energy_kWh": "avg_kwh"})
            .to_dict(orient="records")
        )

        all_outputs[location_id] = {
            "annual_forecast": forecast_df.to_dict(orient="records"),
            "summary": summary,
            "historical_monthly_avg": historical_monthly_avg,
        }

    with open(f"{OUTPUT_DIR}/forecasts_all_locations.json", "w") as f:
        json.dump(all_outputs, f, indent=2)

    print("\n=== Confidence interval widening check (should grow with horizon) ===")
    for location_id, output in all_outputs.items():
        first = output["summary"]["ci_width_first_year_kwh"]
        last = output["summary"]["ci_width_last_year_kwh"]
        print(f"  {location_id}: {first:,.0f} kWh (2027) -> {last:,.0f} kWh (2036)")

    print("\nMilestone 9-10 (Monte Carlo P50/P75/P90) complete.")
    print(f"Per-location CSVs + combined JSON written to {OUTPUT_DIR}/")