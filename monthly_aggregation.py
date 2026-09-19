"""
Milestone 5 / Stage 8.2 — Monthly Aggregation

Aggregates the ~219,000-row hourly series (25 years x ~8766 hrs) per
location down to a compact 300-row monthly series (25 years x 12 months).

Columns: year, month, GHI, DNI, DHI, temperature, humidity, wind_speed,
 POA_irradiance, PV_energy_kWh

"""

import glob
import pandas as pd

from site_configs import PILOT_SITES

PHYSICS_DIR = "data/physics"
MONTHLY_DIR = "data/monthly"

AGG_COLUMNS = {
    "GHI": "mean",
    "DNI": "mean",
    "DHI": "mean",
    "temperature_C": "mean",
    "humidity_pct": "mean",
    "wind_speed_m_s": "mean",
    "POA_irradiance": "mean",
    "P_physical_kWh": "sum",  
}


def aggregate_location_to_monthly(location_id: str) -> pd.DataFrame:
    year_dirs = sorted(glob.glob(f"{PHYSICS_DIR}/location_id={location_id}/year=*"))

    monthly_rows = []
    for year_dir in year_dirs:
        year = int(year_dir.split("year=")[-1])
        df = pd.read_parquet(year_dir)

        grouped = df.groupby("month").agg(AGG_COLUMNS).reset_index()
        grouped["year"] = year
        grouped["location_id"] = location_id
        monthly_rows.append(grouped)

    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    monthly_df = monthly_df.rename(columns={"P_physical_kWh": "PV_energy_kWh"})
    monthly_df = monthly_df.sort_values(["year", "month"]).reset_index(drop=True)

    monthly_df = monthly_df[[
        "location_id", "year", "month", "GHI", "DNI", "DHI",
        "temperature_C", "humidity_pct", "wind_speed_m_s",
        "POA_irradiance", "PV_energy_kWh",
    ]]
    return monthly_df


def process_location(site: dict) -> pd.DataFrame:
    location_id = site["location_id"]
    print(f"\n=== {location_id} ({site['city']}) ===")

    monthly_df = aggregate_location_to_monthly(location_id)

    out_path = f"{MONTHLY_DIR}/{location_id}_monthly.parquet"
    monthly_df.to_parquet(out_path, index=False)

    print(f"  {len(monthly_df)} monthly rows written to {out_path}")
    print(f"  Expected: {len(PILOT_SITES) and 26 * 12} rows "
          f"(26 years x 12 months) -- got {len(monthly_df)}")

    return monthly_df


if __name__ == "__main__":
    import os
    os.makedirs(MONTHLY_DIR, exist_ok=True)

    all_monthly = {}
    for site in PILOT_SITES:
        all_monthly[site["location_id"]] = process_location(site)

    print("\n Sample: first 6 months, Jaisalmer ")
    print(all_monthly["IN_JSL_001"].head(6).to_string(index=False))

    print("\n Annual PV_energy_kWh check (sum of 12 months per year)")
    for location_id, df in all_monthly.items():
        annual = df.groupby("year")["PV_energy_kWh"].sum()
        print(f"  {location_id}: 2001={annual.get(2001, 0):,.0f} kWh, "
              f"2025={annual.get(2025, 0):,.0f} kWh")

    print("\nMilestone 5 (monthly aggregation) complete for all pilot locations.")