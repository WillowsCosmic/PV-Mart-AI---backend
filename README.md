# PV Mart AI — Forecasting Engine (Backend)

A physics-grounded, AI-assisted 10-year solar energy production forecasting
pipeline, built from PV Mart's AI Solar Forecasting Pipeline developer
specification. Given any location in India and a panel/inverter/battery
configuration, it produces a probabilistic P50/P75/P90 forecast per year
for the next decade.

This is a **separate backend service** from the main `PV-Mart-Pro` frontend
repo — Python/FastAPI here, React/Vite there. They communicate over HTTP.

---

## Tech stack

| Layer | Tool |
|---|---|
| Language / package manager | Python 3.12, `uv` |
| Weather data source | NASA POWER API (hourly, 2001–present) |
| Solar geometry / irradiance | `pvlib` |
| Storage (raw + processed data) | Partitioned Parquet, local disk (`data/`) |
| Storage (served results, structured) | TimescaleDB (PostgreSQL), native install |
| ML / statistical models | scikit-learn, XGBoost, LightGBM, statsmodels (SARIMA) |
| Experiment tracking | MLflow (`localhost:5000`) |
| API | FastAPI + Uvicorn |
| DB browsing (optional) | pgAdmin (native install) |

---

## Project structure

```
PV-Mart-AI/
  site_configs.py            Pilot site definitions (lat/lng, tilt, azimuth)
  product_configs.py         Panel/inverter/battery catalog (ported from
                              PV-Mart-Pro's productDatabase.js) + system sizing

  ingest_nasa_power.py       Stage 2 — 26yr hourly NASA POWER ingestion
  solar_geometry.py          Stage 3 — solar position, POA irradiance, cell temp
  loss_model.py              Stage 4 — full loss taxonomy (temp, shading,
                              soiling, wiring, inverter, mismatch, availability,
                              battery, degradation)
  physics_baseline.py        Stage 5 — P_physical baseline output
  monthly_aggregation.py     Stage 8.2 — hourly -> 312-row monthly series

  model_pool.py              Stage 6 — candidate model pool (Persistence,
                              LinearRegression, RandomForest, XGBoost,
                              LightGBM, SARIMA)
  walk_forward.py            Stage 7 / 9 — rolling-origin validation +
                              historical long-horizon (10yr-ahead) simulation
  horizon_error_curve.py     Empirical per-horizon-year error measurement
  pooled_horizon_curve.py    Pools error growth shape across all locations
                              (single-location curves are too noisy alone)
  champion_selection.py      Stage 8.3 — champion/ensemble auto-selection
  monte_carlo_forecast.py    Stage 8 — P50/P75/P90 Monte Carlo forecast
                              (pilot locations only)

  pipeline_orchestrator.py   Refactors the above stages into reusable
                              functions; runs the FULL pipeline for any
                              new coordinate, not just the 3 pilots
  forecast_api.py            FastAPI service — POST /v1/forecast,
                              GET /v1/forecast/status/{job_id}

  data/                      All Parquet output, CSVs, JSON forecast cache
  mlflow.db, mlruns/         MLflow tracking store
```

---

## Setup

### 1. Install dependencies
```bash
uv venv
source .venv/bin/activate
uv pip install pandas pyarrow httpx pvlib scikit-learn xgboost lightgbm \
    statsmodels mlflow fastapi uvicorn
```

### 2. Database — TimescaleDB (native install, not Docker)
```bash
sudo apt install -y timescaledb-2-postgresql-16 postgresql-client-16
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql

sudo -u postgres psql
```
```sql
CREATE DATABASE pvmart;
\c pvmart
CREATE EXTENSION IF NOT EXISTS timescaledb;
ALTER USER postgres PASSWORD 'yourpassword';
```

### 3. MLflow tracking server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```
UI at `http://localhost:5000`.

### 4. Run the pipeline (in order, first time only)
```bash
python ingest_nasa_power.py        # ~75 NASA API calls, several minutes
python solar_geometry.py
python loss_model.py
python physics_baseline.py
python monthly_aggregation.py
python model_pool.py
python walk_forward.py
python horizon_error_curve.py
python pooled_horizon_curve.py
python champion_selection.py
python monte_carlo_forecast.py     # produces data/forecasts_all_locations.json
```
Each script caches its output (Parquet/CSV/JSON) and skips already-processed
years/locations on re-run.

### 5. Run the API
```bash
uvicorn forecast_api:app --reload --port 8000
```
Docs at `http://localhost:8000/docs`.

---

## API

### `POST /v1/forecast`
```json
{
  "site": {"latitude": 26.9157, "longitude": 70.9083, "azimuth_deg": 180},
  "system": {
    "panel": {"model": "Adani Mono 540"},
    "inverter": {"model": "UTL Gamma 5K"},
    "battery": null
  },
  "horizon_years": 10
}
```

- **200** — known pilot site or a previously-processed location: instant result.
- **202** — brand-new coordinate: full pipeline kicked off as a background
  job. Response includes `job_id` and `check_status_url`.

### `GET /v1/forecast/status/{job_id}`
Poll until `"status": "completed"` (or `"failed"`). Completed responses
include the full forecast (`champion_model`, `scoreboard`, `annual_forecast`,
`summary`) — same shape as a 200 from `/v1/forecast`.

---

## Known limitations (honest, not hidden)

- **No real metered generation data yet.** All models train against the
  physics-baseline series (`PV_energy_kWh`), not a residual against real
  SCADA/inverter data — no client has that data available yet. The moment
  it exists, only `TARGET_COLUMN` in `model_pool.py` needs to change.
- **In-memory job queue.** `forecast_api.py`'s job store resets on server
  restart. Fine for single-instance pilot use; a real production deployment
  should move this to TimescaleDB or Celery+Redis so jobs survive restarts
  and work across multiple server instances.
- **New-location pipeline takes several minutes** (25-year NASA pull +
  model training) — this is inherent to the approach, not a bug to fix.
- **Battery support is built but lightly tested** — the loss/degradation
  math is generic and correct, but no pilot site has exercised it against
  real-world numbers yet.
- **Duplicate product model names**: if two catalog entries share a short
  `model` name (rare), `find_by_model()` returns the first match.

---

## The 3 original pilot sites

| Location | ID | Coordinates |
|---|---|---|
| Jaisalmer, RJ | `IN_JSL_001` | 26.9157, 70.9083 |
| Kolkata, WB | `IN_KOL_001` | 22.5726, 88.3639 |
| Pune, MH | `IN_PUN_001` | 18.5204, 73.8567 |

Any other coordinate is supported via the async pipeline (`pipeline_orchestrator.py`).
