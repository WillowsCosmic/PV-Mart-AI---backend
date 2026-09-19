"""
Stage 2 — NASA POWER Data Acquisition (26-Year Historical)

Pulls hourly GHI/DNI/DHI + weather data for each pilot location from the
NASA POWER API, parses into Dataset A schema, flags/interpolates missing
hours, and writes to partitioned Parquet (location_id / year).


"""

import asyncio
import glob
from datetime import datetime, timezone

import httpx
import pandas as pd

from site_configs import PILOT_SITES

# --- Config ---------------------------------------------------------------

NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",  # GHI
    "ALLSKY_SFC_SW_DNI",  # DNI
    "ALLSKY_SFC_SW_DIFF",  # DHI
    "T2M",  # temperature at 2m
    "RH2M",  # relative humidity at 2m
    "WS10M",  # wind speed at 10m
    "PS",  # surface pressure
    "PRECTOTCORR",  # precipitation, corrected
]

START_YEAR = 2001
END_YEAR = datetime.now(timezone.utc).year - 1  # last full year available

OUTPUT_DIR = "data/parquet"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120.0

# NASA POWER hourly endpoint caps how many days you can pull per request;
# pull one year at a time to stay well within limits and to make caching
# per-year straightforward.


def already_cached(location_id: str, year: int) -> bool:
    """Check if this location/year partition already has data on disk."""
    pattern = f"{OUTPUT_DIR}/location_id={location_id}/year={year}/*.parquet"
    return len(glob.glob(pattern)) > 0


async def fetch_year(
    client: httpx.AsyncClient, latitude: float, longitude: float, year: int
) -> dict:
    """Fetch one year of hourly data for one location, with retries."""
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.get(
                NASA_POWER_BASE_URL, params=params, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_exc = exc
            wait = 2**attempt  # exponential backoff: 2s, 4s, 8s, 16s
            print(f"  [retry {attempt}/{MAX_RETRIES}] {year}: {exc} "
                  f"-> waiting {wait}s")
            await asyncio.sleep(wait)

    raise RuntimeError(
        f"Failed to fetch year {year} after {MAX_RETRIES} retries"
    ) from last_exc


def parse_to_dataframe(raw: dict, location_id: str, latitude: float,
                        longitude: float) -> pd.DataFrame:
    """Convert NASA POWER's raw JSON response into Dataset A rows."""
    param_data = raw["properties"]["parameter"]

    # All parameters share the same set of timestamp keys (format: YYYYMMDDHH)
    timestamps = sorted(param_data[PARAMETERS[0]].keys())

    rows = []
    for ts in timestamps:
        dt = datetime.strptime(ts, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        row = {
            "location_id": location_id,
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": dt,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "day_of_year": dt.timetuple().tm_yday,
            "GHI": param_data["ALLSKY_SFC_SW_DWN"].get(ts),
            "DNI": param_data["ALLSKY_SFC_SW_DNI"].get(ts),
            "DHI": param_data["ALLSKY_SFC_SW_DIFF"].get(ts),
            "temperature_C": param_data["T2M"].get(ts),
            "humidity_pct": param_data["RH2M"].get(ts),
            "wind_speed_m_s": param_data["WS10M"].get(ts),
            "pressure_kPa": param_data["PS"].get(ts),
            "precipitation_mm": param_data["PRECTOTCORR"].get(ts),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def flag_and_interpolate(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    NASA POWER uses -999 as a fill value for missing data. Replace with NaN,
    interpolate small gaps, and report the interpolation rate as a
    data-quality metric.
    """
    value_cols = [
        "GHI", "DNI", "DHI", "temperature_C", "humidity_pct",
        "wind_speed_m_s", "pressure_kPa", "precipitation_mm",
    ]

    df = df.sort_values("timestamp").reset_index(drop=True)
    df[value_cols] = df[value_cols].replace(-999, pd.NA)

    total_cells = len(df) * len(value_cols)
    missing_before = df[value_cols].isna().sum().sum()

    # Linear interpolation for short gaps (time-ordered, so this is safe —
    # no shuffling has happened at this point in the pipeline)
    df[value_cols] = df[value_cols].interpolate(method="linear", limit=6)

    missing_after = df[value_cols].isna().sum().sum()
    interpolation_rate = (
        (missing_before - missing_after) / total_cells if total_cells else 0.0
    )

    return df, interpolation_rate


async def ingest_location(client: httpx.AsyncClient, site: dict) -> None:
    location_id = site["location_id"]
    lat, lon = site["latitude"], site["longitude"]

    print(f"\n=== {location_id} ({site['city']}) ===")

    for year in range(START_YEAR, END_YEAR + 1):
        if already_cached(location_id, year):
            print(f"  {year}: already cached, skipping")
            continue

        print(f"  {year}: fetching...")
        raw = await fetch_year(client, lat, lon, year)
        df = parse_to_dataframe(raw, location_id, lat, lon)
        df, interp_rate = flag_and_interpolate(df)

        df.to_parquet(
            OUTPUT_DIR,
            partition_cols=["location_id", "year"],
            index=False,
        )

        print(f"  {year}: {len(df)} rows written, "
              f"interpolation rate = {interp_rate:.4%}")


async def main():
    async with httpx.AsyncClient() as client:
        for site in PILOT_SITES:
            await ingest_location(client, site)

    print("\nIngestion complete for all pilot locations.")


if __name__ == "__main__":
    asyncio.run(main())