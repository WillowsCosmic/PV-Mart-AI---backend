"""
Backend Phase 2 — Any-Location Pipeline Orchestrator

Runs the ENTIRE pipeline (Stages 1-8 / Milestones 1-10) for a single,
arbitrary new coordinate -- not just the 3 hardcoded pilot sites.

This is what makes the frontend's "every location" requirement possible:
instead of only serving pre-computed results, the API can now trigger
this function for any lat/lng, wait for it to finish, and cache the
result exactly like a new pilot site.

Because this takes several minutes (26-year hourly NASA pull + model
training), it's designed to be called from a background job (see
forecast_api.py's async job queue), not synchronously in a request.

Run standalone for testing:
    python pipeline_orchestrator.py
"""

import glob
import os
import hashlib
import warnings
import numpy as np
import pandas as pd

import httpx
import asyncio
import pvlib

from model_pool import (
    add_lag_features, compute_metrics, CANDIDATES, TARGET_COLUMN,
)
from loss_model import (
    temperature_loss_factor, shading_loss_factor, soiling_loss_factor,
    inverter_loss_factor, module_degradation_factor,
    battery_round_trip_loss_factor, battery_degradation_factor,
    DC_WIRING_LOSS_PCT, AC_WIRING_LOSS_PCT, MISMATCH_LOSS_PCT,
    AVAILABILITY_LOSS_PCT,
)
from ingest_nasa_power import (
    START_YEAR, END_YEAR, fetch_year, parse_to_dataframe,
    flag_and_interpolate, already_cached,
)

warnings.filterwarnings("ignore")

INSTALL_YEAR = 2027
DEGRADATION_RATE_RANGE = (0.004, 0.006)
SOILING_UNCERTAINTY = (-0.01, 0.01)
N_SIMULATIONS = 1000
FORECAST_START_YEAR = 2027
FORECAST_YEARS = 10
VALIDATION_TEST_YEARS = 3
NOISE_BAND_PCT = 0.05

# The pooled horizon-error growth SHAPE (from pooled_horizon_curve.py) is
# reused for any new location -- it's a cross-location statistical
# pattern, not specific to the 3 pilots. Only the location's OWN baseline
# (year-1) error is measured fresh for each new site.
POOLED_SHAPE = pd.read_csv("data/horizon_error_curve_final.csv")
_shape_lookup = POOLED_SHAPE[POOLED_SHAPE["location_id"] == "IN_JSL_001"].copy()
_shape_lookup["multiplier"] = _shape_lookup["nrmse"] / _shape_lookup.iloc[0]["nrmse"]
POOLED_MULTIPLIER = dict(zip(_shape_lookup["horizon_year"], _shape_lookup["multiplier"]))


def make_location_id(latitude: float, longitude: float) -> str:
    """Deterministic ID from coordinates (rounded to ~11m precision) --
    requesting the same site twice reuses the cached result instead of
    re-running the whole pipeline."""
    rounded = f"{round(latitude, 4)}_{round(longitude, 4)}"
    digest = hashlib.md5(rounded.encode()).hexdigest()[:8]
    return f"USER_{digest}"


# --- Stage 2: ingestion for one new location --------------------------------

async def ingest_location(location_id: str, latitude: float, longitude: float) -> None:
    async with httpx.AsyncClient() as client:
        for year in range(START_YEAR, END_YEAR + 1):
            if already_cached(location_id, year):
                continue
            raw = await fetch_year(client, latitude, longitude, year)
            df = parse_to_dataframe(raw, location_id, latitude, longitude)
            df, _ = flag_and_interpolate(df)
            df.to_parquet("data/parquet", partition_cols=["location_id", "year"], index=False)


# --- Stage 3: solar geometry for one new location ----------------------------

def compute_geometry(location_id: str, latitude: float, longitude: float,
                      tilt_deg: float, azimuth_deg: float) -> None:
    year_dirs = sorted(glob.glob(f"data/parquet/location_id={location_id}/year=*"))
    for year_dir in year_dirs:
        year = int(year_dir.split("year=")[-1])
        if glob.glob(f"data/features/location_id={location_id}/year={year}/*.parquet"):
            continue

        df = pd.read_parquet(year_dir)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["location_id"], df["year"] = location_id, year

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

        times = pd.DatetimeIndex(df["timestamp"])
        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        df["solar_zenith"] = solpos["zenith"].values
        df["solar_azimuth"] = solpos["azimuth"].values
        df["solar_elevation"] = solpos["elevation"].values

        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt_deg, surface_azimuth=azimuth_deg,
            solar_zenith=df["solar_zenith"].values, solar_azimuth=df["solar_azimuth"].values,
            dni=df["DNI"].values, ghi=df["GHI"].values, dhi=df["DHI"].values,
        )
        df["POA_irradiance"] = np.asarray(poa["poa_global"])
        df["cell_temperature"] = pvlib.temperature.faiman(
            df["POA_irradiance"].values, df["temperature_C"].values, df["wind_speed_m_s"].values,
        )
        df.to_parquet("data/features", partition_cols=["location_id", "year"], index=False)


# --- Stage 4: loss model for one new location --------------------------------

def compute_losses(location_id: str, panel: dict, inverter: dict, battery: dict | None) -> None:
    year_dirs = sorted(glob.glob(f"data/features/location_id={location_id}/year=*"))
    for year_dir in year_dirs:
        year = int(year_dir.split("year=")[-1])
        if glob.glob(f"data/losses/location_id={location_id}/year={year}/*.parquet"):
            continue

        df = pd.read_parquet(year_dir)
        df["location_id"], df["year"] = location_id, year

        df["loss_temp"] = temperature_loss_factor(df["cell_temperature"].values, panel["gamma"])
        df["loss_shading"] = shading_loss_factor([], df["month"].values)
        df["loss_soiling"] = soiling_loss_factor(df["day_of_year"].values, None)
        df["loss_dc"] = 1 - DC_WIRING_LOSS_PCT
        df["loss_ac"] = 1 - AC_WIRING_LOSS_PCT
        df["loss_mismatch"] = 1 - MISMATCH_LOSS_PCT
        df["loss_availability"] = 1 - AVAILABILITY_LOSS_PCT

        poa_fraction = np.clip(df["POA_irradiance"].values / 1000.0, 0, 1.2)
        df["loss_inverter"] = inverter_loss_factor(poa_fraction, inverter["eff"])
        df["loss_battery_rt"] = battery_round_trip_loss_factor(battery)

        degr = battery_degradation_factor(battery, year, INSTALL_YEAR)
        df["degradation_factor"] = module_degradation_factor(year, INSTALL_YEAR) * degr

        df["eta_losses"] = (
            df["loss_temp"] * df["loss_shading"] * df["loss_soiling"] * df["loss_dc"] *
            df["loss_ac"] * df["loss_mismatch"] * df["loss_availability"] *
            df["loss_inverter"] * df["loss_battery_rt"] * df["degradation_factor"]
        )
        df.to_parquet("data/losses", partition_cols=["location_id", "year"], index=False)


# --- Stage 5: physics baseline + system sizing -------------------------------

def compute_system_size(panel: dict, inverter: dict, dc_ac_ratio: float = 1.15) -> dict:
    target_dc_kw = inverter["kw"] * dc_ac_ratio
    num_panels = round((target_dc_kw * 1000) / panel["watt"])
    return {"num_panels": num_panels, "system_kwp": (num_panels * panel["watt"]) / 1000}


def compute_physics_baseline(location_id: str, panel: dict, num_panels: int) -> None:
    p_stc_kw = (panel["watt"] * num_panels) / 1000.0
    year_dirs = sorted(glob.glob(f"data/losses/location_id={location_id}/year=*"))
    for year_dir in year_dirs:
        year = int(year_dir.split("year=")[-1])
        if glob.glob(f"data/physics/location_id={location_id}/year={year}/*.parquet"):
            continue
        df = pd.read_parquet(year_dir)
        df["location_id"], df["year"] = location_id, year
        df["P_physical_kW"] = (p_stc_kw * (df["POA_irradiance"] / 1000.0) * df["eta_losses"]).clip(lower=0)
        df["P_physical_kWh"] = df["P_physical_kW"]
        df.to_parquet("data/physics", partition_cols=["location_id", "year"], index=False)


# --- Stage 8.2: monthly aggregation ------------------------------------------

def aggregate_monthly(location_id: str) -> pd.DataFrame:
    year_dirs = sorted(glob.glob(f"data/physics/location_id={location_id}/year=*"))
    agg = {"GHI": "mean", "DNI": "mean", "DHI": "mean", "temperature_C": "mean",
           "humidity_pct": "mean", "wind_speed_m_s": "mean", "POA_irradiance": "mean",
           "P_physical_kWh": "sum"}
    rows = []
    for year_dir in year_dirs:
        year = int(year_dir.split("year=")[-1])
        df = pd.read_parquet(year_dir)
        grouped = df.groupby("month").agg(agg).reset_index()
        grouped["year"], grouped["location_id"] = year, location_id
        rows.append(grouped)
    monthly = pd.concat(rows, ignore_index=True).rename(columns={"P_physical_kWh": "PV_energy_kWh"})
    monthly = monthly.sort_values(["year", "month"]).reset_index(drop=True)
    os.makedirs("data/monthly", exist_ok=True)
    monthly.to_parquet(f"data/monthly/{location_id}_monthly.parquet", index=False)
    return monthly


# --- Stage 6-8: champion selection + Monte Carlo -----------------------------

def select_champion(location_id: str, monthly_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = add_lag_features(monthly_df)
    cutoff = df["year"].max() - VALIDATION_TEST_YEARS + 1
    train, val = df[df["year"] < cutoff], df[df["year"] >= cutoff]
    y_true = val[TARGET_COLUMN].values

    scoreboard, predictions = [], {}
    for name, fn in CANDIDATES.items():
        try:
            y_pred = fn(train, val)
            metrics = compute_metrics(y_true, y_pred)
            scoreboard.append({"model": name, **metrics})
            predictions[name] = y_pred
        except Exception:
            continue

    scoreboard_df = pd.DataFrame(scoreboard).sort_values("nrmse").reset_index(drop=True)
    best = scoreboard_df.iloc[0]["nrmse"]
    within_band = scoreboard_df[scoreboard_df["nrmse"] <= best * (1 + NOISE_BAND_PCT)]

    if len(within_band) > 1:
        champion_model = " + ".join(within_band["model"].tolist())
    else:
        champion_model = scoreboard_df.iloc[0]["model"]

    year1_nrmse = float(scoreboard_df.iloc[0]["nrmse"])
    return {"champion_model": champion_model, "year1_nrmse": year1_nrmse}, scoreboard_df


def run_monte_carlo(monthly_df: pd.DataFrame, year1_nrmse: float, system_kwp: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    annual = monthly_df.groupby("year")["PV_energy_kWh"].sum()
    baseline_mean, baseline_std = annual.mean(), annual.std()

    trajectories = np.zeros((N_SIMULATIONS, FORECAST_YEARS))
    for sim in range(N_SIMULATIONS):
        degradation_rate = rng.uniform(*DEGRADATION_RATE_RANGE)
        soiling_offset = rng.uniform(*SOILING_UNCERTAINTY)
        for year_idx in range(FORECAST_YEARS):
            horizon_year = year_idx + 1
            multiplier = POOLED_MULTIPLIER.get(horizon_year, 1.0)
            nrmse_h = year1_nrmse * multiplier

            resource_draw = rng.normal(baseline_mean, baseline_std)
            model_error = rng.normal(0, baseline_mean * nrmse_h)
            degradation_factor = (1 - degradation_rate) ** year_idx
            soiling_factor = 1 - soiling_offset

            annual_kwh = (resource_draw + model_error) * degradation_factor * soiling_factor
            trajectories[sim, year_idx] = max(0, annual_kwh)

    rows = []
    for year_idx in range(FORECAST_YEARS):
        col = trajectories[:, year_idx]
        rows.append({
            "year": FORECAST_START_YEAR + year_idx,
            "p50_kwh": round(np.percentile(col, 50), 1),
            "p75_kwh": round(np.percentile(col, 25), 1),
            "p90_kwh": round(np.percentile(col, 10), 1),
        })
    return pd.DataFrame(rows)


# --- Orchestration entry point -----------------------------------------------

async def run_full_pipeline_for_new_site(site: dict, system: dict) -> dict:
    """The single function the API calls for any new coordinate. Runs
    Stages 2-8 end to end and returns a Section-14-shaped forecast dict."""
    latitude, longitude = site["latitude"], site["longitude"]
    location_id = make_location_id(latitude, longitude)
    tilt_deg = site.get("tilt_deg") or round(0.76 * abs(latitude), 2)
    azimuth_deg = site.get("azimuth_deg", 180.0)

    panel, inverter, battery = system["panel"], system["inverter"], system.get("battery")

    await ingest_location(location_id, latitude, longitude)
    compute_geometry(location_id, latitude, longitude, tilt_deg, azimuth_deg)
    compute_losses(location_id, panel, inverter, battery)

    sizing = compute_system_size(panel, inverter)
    compute_physics_baseline(location_id, panel, sizing["num_panels"])

    monthly_df = aggregate_monthly(location_id)
    champion_decision, scoreboard_df = select_champion(location_id, monthly_df)
    forecast_df = run_monte_carlo(monthly_df, champion_decision["year1_nrmse"], sizing["system_kwp"])

    total_10yr_kwh = forecast_df["p50_kwh"].sum()
    avg_annual = forecast_df["p50_kwh"].mean()

    historical_monthly_avg = (
        monthly_df.groupby("month")["PV_energy_kWh"]
        .mean().round(1).reset_index()
        .rename(columns={"PV_energy_kWh": "avg_kwh"})
        .to_dict(orient="records")
    )
    historical_monthly_full = (
        monthly_df[["year", "month", "PV_energy_kWh"]]
        .sort_values(["year", "month"])
        .round(1)
        .rename(columns={"PV_energy_kWh": "kwh"})
        .to_dict(orient="records")
    )

    return {
        "location_id": location_id,
        "champion_model": champion_decision["champion_model"],
        "scoreboard": scoreboard_df.round(4).to_dict(orient="records"),
        "annual_forecast": forecast_df.to_dict(orient="records"),
        "historical_monthly_avg": historical_monthly_avg,
        "historical_monthly_full": historical_monthly_full,
        "summary": {
            "total_10yr_mwh": round(total_10yr_kwh / 1000, 2),
            "avg_annual_kwh": round(avg_annual, 1),
            "specific_yield_kwh_per_kwp": round(avg_annual / sizing["system_kwp"], 1),
            "cuf_pct": round((avg_annual / (sizing["system_kwp"] * 8760)) * 100, 2),
            "system_kwp": sizing["system_kwp"],
            "num_panels": sizing["num_panels"],
        },
    }


if __name__ == "__main__":
    # Quick standalone test with a brand-new coordinate (Ahmedabad),
    # not one of the 3 original pilots.
    from products_config import find_by_model, PANELS, INVERTERS

    test_site = {"latitude": 23.0225, "longitude": 72.5714, "azimuth_deg": 180.0}
    test_system = {
        "panel": find_by_model(PANELS, "Adani Mono 540"),
        "inverter": find_by_model(INVERTERS, "UTL Gamma 5K"),
        "battery": None,
    }

    result = asyncio.run(run_full_pipeline_for_new_site(test_site, test_system))
    print(f"\nNew location processed: {result['location_id']}")
    print(f"Champion: {result['champion_model']}")
    print(f"Summary: {result['summary']}")