"""
Stage 3 — Feature Engineering & Solar Geometry

Reads the raw hourly weather data written by ingest_nasa_power.py, computes:
  - cyclic time features (hour/month/day_of_year sin/cos)
  - solar position (zenith, elevation, azimuth) via pvlib
  - plane-of-array (POA) irradiance (GHI/DNI/DHI -> tilted panel plane)
  - estimated cell temperature (Sandia/Faiman-style, via pvlib)

Writes the enriched dataset to data/features/ (same partitioning scheme:
location_id / year), ready to feed Stage 4 (loss model) and Stage 5
(physics baseline).

"""

import glob
import numpy as np
import pandas as pd
import pvlib

from site_configs import PILOT_SITES, get_effective_tilt

RAW_DIR = "data/parquet"
FEATURES_DIR = "data/features"


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_solar_position_and_poa(
    df: pd.DataFrame, latitude: float, longitude: float, tilt_deg: float,
    azimuth_deg: float,
) -> pd.DataFrame:
    times = pd.DatetimeIndex(df["timestamp"]).tz_localize("UTC") \
        if df["timestamp"].dt.tz is None else pd.DatetimeIndex(df["timestamp"])

    solpos = pvlib.solarposition.get_solarposition(
        time=times, latitude=latitude, longitude=longitude
    )
    df["solar_zenith"] = solpos["zenith"].values
    df["solar_azimuth"] = solpos["azimuth"].values
    df["solar_elevation"] = solpos["elevation"].values

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=df["solar_zenith"].values,
        solar_azimuth=df["solar_azimuth"].values,
        dni=df["DNI"].values,
        ghi=df["GHI"].values,
        dhi=df["DHI"].values,
    )
    df["POA_irradiance"] = np.asarray(poa["poa_global"])

    return df


def add_cell_temperature(df: pd.DataFrame) -> pd.DataFrame:


    df["cell_temperature"] = pvlib.temperature.faiman(
        poa_global=df["POA_irradiance"].values,
        temp_air=df["temperature_C"].values,
        wind_speed=df["wind_speed_m_s"].values,
    )
    return df


def process_location(site: dict) -> None:
    location_id = site["location_id"]
    tilt = get_effective_tilt(site)
    azimuth = site["azimuth_deg"]

    print(f"\n {location_id} ({site['city']}) — tilt={tilt}, azimuth={azimuth} ")

    year_dirs = sorted(glob.glob(f"{RAW_DIR}/location_id={location_id}/year=*"))

    for year_dir in year_dirs:
        year = int(year_dir.split("year=")[-1])

        out_check = glob.glob(f"{FEATURES_DIR}/location_id={location_id}/year={year}/*.parquet")
        if out_check:
            print(f"  {year}: already processed, skipping")
            continue

        df = pd.read_parquet(year_dir)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # re-attach partition columns lost on a single-partition read
        df["location_id"] = location_id
        df["year"] = year

        df = add_cyclic_time_features(df)
        df = add_solar_position_and_poa(
            df, site["latitude"], site["longitude"], tilt, azimuth
        )
        df = add_cell_temperature(df)

        df.to_parquet(
            FEATURES_DIR,
            partition_cols=["location_id", "year"],
            index=False,
        )

        print(f"  {year}: {len(df)} rows, "
              f"POA max={df['POA_irradiance'].max():.1f} W/m2, "
              f"cell_temp max={df['cell_temperature'].max():.1f} C")


def validate_against_pvlib_reference() -> None:
    

    test_time = pd.DatetimeIndex(["2024-06-21 07:00:00"], tz="UTC") 
    solpos = pvlib.solarposition.get_solarposition(
        test_time, latitude=26.9157, longitude=70.9083
    )
    print("Jaisalmer, summer solstice ~solar noon:")
    print(solpos[["zenith", "elevation", "azimuth"]])
    print("Expected: elevation should be near this location's max for the "
          "year (~86-87 deg, since Jaisalmer's latitude ~27N is close to "
          "the solstice sun's declination of ~23.4N).")


if __name__ == "__main__":
    validate_against_pvlib_reference()

    for site in PILOT_SITES:
        process_location(site)

    print("\nStage 3 (solar geometry + POA) complete for all pilot locations.")