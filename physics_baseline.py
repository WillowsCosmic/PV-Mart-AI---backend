"""
Stage 5 — Physics-Based PV Baseline Model

Computes a deterministic physical estimate of PV output, before any AI
touches it:

    P_physical = P_STC * (POA / 1000) * eta_losses

This provides:
  (a) a sanity-checkable floor for every forecast, and
  (b) the target the AI layer (Stage 6) will learn to correct, via
      Error = P_actual - P_physical (once metered data exists).

"""

import glob
import pandas as pd

from site_configs import PILOT_SITES
from products_config import SITE_SYSTEM_CONFIG

LOSSES_DIR = "data/losses"
PHYSICS_DIR = "data/physics"


def compute_p_physical(df: pd.DataFrame, panel: dict, num_panels: int) -> pd.DataFrame:
    p_stc_kw = (panel["watt"] * num_panels) / 1000.0  # total array rated power, kW

    # P_physical in kW at each hour; since data is hourly, kW == kWh for that hour
    df["P_physical_kW"] = p_stc_kw * (df["POA_irradiance"] / 1000.0) * df["eta_losses"]
    df["P_physical_kW"] = df["P_physical_kW"].clip(lower=0)  # no negative generation
    df["P_physical_kWh"] = df["P_physical_kW"]  # 1-hour resolution
    return df


def process_location(site: dict) -> dict:
    location_id = site["location_id"]
    system = SITE_SYSTEM_CONFIG[location_id]
    panel, num_panels, system_kwp = system["panel"], system["num_panels"], system["system_kwp"]

    print(f"\n=== {location_id} ({site['city']}) — "
          f"{num_panels} x {panel['model']} = {system_kwp:.2f} kWp ===")

    year_dirs = sorted(glob.glob(f"{LOSSES_DIR}/location_id={location_id}/year=*"))
    annual_totals = {}

    for year_dir in year_dirs:
        year = int(year_dir.split("year=")[-1])

        out_check = glob.glob(f"{PHYSICS_DIR}/location_id={location_id}/year={year}/*.parquet")
        if out_check:
            df = pd.read_parquet(out_check[0])
        else:
            df = pd.read_parquet(year_dir)
            df["location_id"] = location_id
            df["year"] = year
            df = compute_p_physical(df, panel, num_panels)
            df.to_parquet(PHYSICS_DIR, partition_cols=["location_id", "year"], index=False)

        annual_kwh = df["P_physical_kWh"].sum()
        specific_yield = annual_kwh / system_kwp  # kWh/kWp/yr
        annual_totals[year] = annual_kwh

        print(f"  {year}: {annual_kwh:,.0f} kWh  |  "
              f"specific yield = {specific_yield:,.0f} kWh/kWp  |  "
              f"CUF = {(annual_kwh / (system_kwp * 8760)) * 100:.1f}%")

    return annual_totals


if __name__ == "__main__":
    all_results = {}
    for site in PILOT_SITES:
        all_results[site["location_id"]] = process_location(site)

    print("\n=== Cross-location summary (avg annual kWh across 2001-2025) ===")
    for location_id, totals in all_results.items():
        avg = sum(totals.values()) / len(totals)
        print(f"  {location_id}: {avg:,.0f} kWh/yr average")

    print("\nStage 5 (physics baseline) complete for all pilot locations.")