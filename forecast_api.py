"""
Backend Phase 2 — FastAPI service, now supporting ANY location.

Behavior:
  - Coordinates matching one of the 3 known pilot sites -> instant response
    (pre-computed, same as before).
  - Coordinates matching a previously-processed NEW location -> instant
    response (cached from a prior run).
  - A genuinely new coordinate -> kicks off the full pipeline as a
    background job, returns 202 with a job_id immediately. Frontend polls
    GET /v1/forecast/status/{job_id} until it's done (several minutes).

Run:
    uvicorn forecast_api:app --reload --port 8000
"""

import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pipeline_orchestrator import make_location_id, run_full_pipeline_for_new_site
from products_config import find_by_model, PANELS, INVERTERS, BATTERIES

FORECASTS_PATH = Path("data/forecasts_all_locations.json")
NEW_LOCATIONS_CACHE_PATH = Path("data/forecasts_new_locations.json")

PILOT_COORDINATES = {
    "IN_JSL_001": (26.9157, 70.9083),
    "IN_KOL_001": (22.5726, 88.3639),
    "IN_PUN_001": (18.5204, 73.8567),
}

app = FastAPI(title="PV Mart AI Forecasting Engine", version="0.2.0-any-location")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5174"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# In-memory job store. NOTE: this resets if the server restarts -- fine
# for a single-instance pilot deployment. A real production version
# would persist this in TimescaleDB or use Celery+Redis so jobs survive
# restarts and work across multiple server instances.
job_store: dict[str, dict] = {}


class SiteInput(BaseModel):
    latitude: float
    longitude: float
    tilt_deg: Optional[float] = None
    azimuth_deg: float = 180.0
    obstacles: list = Field(default_factory=list)


class SystemInput(BaseModel):
    panel: dict
    inverter: dict
    battery: Optional[dict] = None


class ForecastRequest(BaseModel):
    site: SiteInput
    system: SystemInput
    losses: Optional[dict] = None
    horizon_years: int = 10


def find_nearest_pilot_location(latitude: float, longitude: float,
                                  tolerance_deg: float = 0.5) -> Optional[str]:
    for location_id, (lat, lon) in PILOT_COORDINATES.items():
        if abs(lat - latitude) < tolerance_deg and abs(lon - longitude) < tolerance_deg:
            return location_id
    return None


def load_json_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_to_new_location_cache(location_id: str, result: dict) -> None:
    cache = load_json_cache(NEW_LOCATIONS_CACHE_PATH)
    cache[location_id] = result
    NEW_LOCATIONS_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def resolve_product(product_input: dict, catalog: list) -> dict:
    """Accepts either a full product spec dict, or {"model": "..."} to
    look up from the real product catalog."""
    if product_input and set(product_input.keys()) == {"model"}:
        return find_by_model(catalog, product_input["model"])
    return product_input


def get_pilot_champion_model(location_id: str) -> str:
    try:
        import pandas as pd
        decisions = pd.read_csv("data/champion_decisions_m8.csv")
        row = decisions[decisions["location_id"] == location_id]
        if not row.empty:
            return row.iloc[0]["champion_model"]
    except Exception:
        pass
    return "unknown"


def get_pilot_scoreboard(location_id: str) -> list:
    try:
        import pandas as pd
        scoreboard = pd.read_csv("data/scoreboard_m7_folds.csv")
        subset = scoreboard[scoreboard["location_id"] == location_id]
        avg = subset.groupby("model")[["nrmse", "smape", "r2"]].mean().reset_index()
        return avg.round(4).to_dict(orient="records")
    except Exception:
        return []


def shape_response(result: dict) -> dict:
    return {
        "champion_model": result["champion_model"],
        "scoreboard": result["scoreboard"],
        "annual_forecast": result["annual_forecast"],
        "summary": result["summary"],
    }


async def _run_pipeline_job(job_id: str, site: dict, system: dict) -> None:
    job_store[job_id]["status"] = "running"
    try:
        result = await run_full_pipeline_for_new_site(site, system)
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["result"] = result
        save_to_new_location_cache(result["location_id"], result)
    except Exception as exc:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(exc)


def find_running_job_for_location(location_id: str) -> Optional[str]:
    for job_id, job in job_store.items():
        if job.get("location_id") == location_id and job["status"] in ("pending", "running"):
            return job_id
    return None


@app.get("/")
def root():
    return {
        "service": "PV Mart AI Forecasting Engine",
        "status": "running",
        "pilot_locations": list(PILOT_COORDINATES.keys()),
        "mode": "any-location (async pipeline for new coordinates)",
    }


@app.post("/v1/forecast")
async def forecast(request: ForecastRequest):
    latitude, longitude = request.site.latitude, request.site.longitude

    # 1. Known pilot site -> instant, pre-computed response
    pilot_id = find_nearest_pilot_location(latitude, longitude)
    if pilot_id:
        all_forecasts = load_json_cache(FORECASTS_PATH)
        if pilot_id in all_forecasts:
            pilot_result = dict(all_forecasts[pilot_id])
            pilot_result["champion_model"] = get_pilot_champion_model(pilot_id)
            pilot_result["scoreboard"] = get_pilot_scoreboard(pilot_id)
            return shape_response(pilot_result)

    # 2. Previously-processed new location -> instant, cached response
    location_id = make_location_id(latitude, longitude)
    new_cache = load_json_cache(NEW_LOCATIONS_CACHE_PATH)
    if location_id in new_cache:
        return shape_response(new_cache[location_id])

    # 3. Already running for this location -> point at the existing job
    existing_job = find_running_job_for_location(location_id)
    if existing_job:
        return JSONResponse(status_code=202, content={
            "status": job_store[existing_job]["status"],
            "job_id": existing_job,
            "check_status_url": f"/v1/forecast/status/{existing_job}",
        })

    # 4. Genuinely new location -> resolve products, kick off background job
    panel = resolve_product(request.system.panel, PANELS)
    inverter = resolve_product(request.system.inverter, INVERTERS)
    battery = resolve_product(request.system.battery, BATTERIES) if request.system.battery else None

    site_dict = {
        "latitude": latitude, "longitude": longitude,
        "tilt_deg": request.site.tilt_deg, "azimuth_deg": request.site.azimuth_deg,
    }
    system_dict = {"panel": panel, "inverter": inverter, "battery": battery}

    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "pending", "location_id": location_id}
    asyncio.create_task(_run_pipeline_job(job_id, site_dict, system_dict))

    return JSONResponse(status_code=202, content={
        "status": "processing",
        "job_id": job_id,
        "location_id": location_id,
        "check_status_url": f"/v1/forecast/status/{job_id}",
        "note": "New location -- running full pipeline (several minutes). Poll the status URL.",
    })


@app.get("/v1/forecast/status/{job_id}")
def forecast_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    job = job_store[job_id]
    if job["status"] == "completed":
        return {"status": "completed", **shape_response(job["result"])}
    if job["status"] == "failed":
        return {"status": "failed", "error": job.get("error", "unknown error")}
    return {"status": job["status"]}