# PV Mart AI — Forecasting Engine (Backend)

A physics-grounded, AI-assisted 10-year solar energy production forecasting
pipeline, built from PV Mart's AI Solar Forecasting Pipeline developer
specification. Given any location in India and a panel/inverter/battery
configuration, it produces a probabilistic P50/P75/P90 forecast per year
for the next decade.

This is a **separate backend service** from the main `PV-Mart-Pro` frontend
repo — Python/FastAPI here, React/Vite there.

**🟢 Live:** https://pv-mart-ai-backend.onrender.com
Companion frontend: https://pv-mart-ai-frontend.vercel.app

---

## Tech stack

| Layer | Tool |
|---|---|
| Language / package manager (local dev) | Python 3.12, `uv` |
| Weather data source | NASA POWER API (hourly, 2001–present) |
| Solar geometry / irradiance | `pvlib` |
| Storage (raw + processed data) | Partitioned Parquet, local disk (`data/`) |
| Storage (structured, served data) | TimescaleDB — local dev only, not yet used in production |
| ML / statistical models | scikit-learn, XGBoost, LightGBM, statsmodels (SARIMA, ETS) |
| Experiment tracking (local dev only) | MLflow (`localhost:5000`) |
| API | FastAPI + Uvicorn |
| **Hosting** | **Render (free tier)** |
| **Keep-alive** | **External cron (cron-job.org) pinging `/health` every 10 min** |

---

## Project structure

```
PV-Mart-AI/
  site_configs.py            Pilot site definitions (lat/lng, tilt, azimuth)
  product_configs.py         Full panel/inverter/battery catalog (ported from
                              PV-Mart-Pro's productDatabase.js) + system sizing

  ingest_nasa_power.py       Stage 2 — 26yr hourly NASA POWER ingestion
  solar_geometry.py          Stage 3 — solar position, POA irradiance, cell temp
  loss_model.py              Stage 4 — full loss taxonomy
  physics_baseline.py        Stage 5 — P_physical baseline output
  monthly_aggregation.py     Stage 8.2 — hourly -> ~300-row monthly series

  model_pool.py              Stage 6 — candidate pool: Persistence,
                              LinearRegression, RandomForest, XGBoost,
                              LightGBM, SARIMA, ETS. Feature set includes
                              lag_1/2/3/12, rolling means (3/6/12mo), and
                              two seasonal harmonics (captures India's
                              bimodal pre/post-monsoon pattern).
                              NOTE: `mlflow` is imported lazily inside
                              `if __name__ == "__main__"` only — a
                              top-level import broke the Render deploy
                              the first time around (mlflow is dev-only,
                              deliberately excluded from requirements.txt).
  walk_forward.py            Stage 7 / 9 — rolling-origin validation +
                              historical long-horizon (10yr-ahead) simulation
  horizon_error_curve.py     Empirical per-horizon-year error measurement
  pooled_horizon_curve.py    Pools error growth SHAPE across all locations
  champion_selection.py      Stage 8.3 — picks the champion DIRECTLY from
                              data/scoreboard_m7_folds.csv's averaged
                              walk-forward numbers (the same numbers the
                              frontend displays), so the champion badge
                              can never contradict the visible scoreboard.
  monte_carlo_forecast.py    Stage 8 — P50/P75/P90 Monte Carlo forecast for
                              the 3 pilots, plus historical_monthly_avg
                              (12-row) and historical_monthly_full
                              (~300-row) series for the export feature.
                              NOTE: earlier had a duplicate typo-named file
                              (Monte_carlo_forcast.py) — consolidated to
                              this single correctly-spelled file.
  pooled_model.py            Experiment (not deployed): tried training one
                              shared model across all 3 locations instead
                              of per-location models. Made things WORSE at
                              every site — kept as a documented negative
                              result, not wired into the live pipeline.

  pipeline_orchestrator.py   Runs the FULL pipeline (Stages 2-8) for any
                              NEW coordinate, not just the 3 pilots.
                              Creates its own working directories on first
                              use (required on a fresh deploy, which has
                              none of these yet). Returns the same fields
                              as monte_carlo_forecast.py's pilot output.
  forecast_api.py            FastAPI service. Config (DATA_DIR,
                              ALLOWED_ORIGINS) read from environment
                              variables. CORS allows both an exact
                              ALLOWED_ORIGINS list and any *.vercel.app
                              origin (via allow_origin_regex), since
                              Vercel generates a new preview URL per
                              deploy that an exact match wouldn't cover.

  requirements.txt           Pip deps for Render's build (deliberately
                              excludes mlflow — dev-only)
  Procfile                   uvicorn ... --port $PORT

  data/                      Parquet output (gitignored, regenerated at
                              runtime), plus 4 COMMITTED files the
                              deployed API depends on directly
  mlflow.db, mlruns/         MLflow tracking store (local dev only, gitignored)
```

---

## Local development setup

```bash
uv venv && source .venv/bin/activate
uv pip install pandas pyarrow httpx pvlib scikit-learn xgboost lightgbm \
    statsmodels mlflow fastapi uvicorn

# TimescaleDB (native, not Docker) -- optional for local dev, unused in prod
sudo apt install -y timescaledb-2-postgresql-16 postgresql-client-16
sudo timescaledb-tune --quiet --yes && sudo systemctl restart postgresql

# MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Run the pipeline once, in order, to generate all working data:
```bash
python ingest_nasa_power.py
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

Run the API locally:
```bash
uvicorn forecast_api:app --reload --port 8000
```
Docs: `http://localhost:8000/docs`

---

## Deployment (Render, free tier) — currently live

### Required committed data files
Render's free tier has **ephemeral disk** — anything written at runtime is
lost on restart. These files must exist in the git repo (not gitignored):
```
data/forecasts_all_locations.json
data/champion_decisions_m8.csv
data/scoreboard_m7_folds.csv
data/horizon_error_curve_final.csv   <- loaded at import time; missing this
                                          crashes the app on startup
```

### Steps taken
1. GitHub repo connected to Render → New Web Service
2. Build: `pip install -r requirements.txt`
3. Start: `uvicorn forecast_api:app --host 0.0.0.0 --port $PORT`
4. Environment variable: `ALLOWED_ORIGINS=https://pv-mart-ai-frontend.vercel.app`
5. Free instance type

### Issues hit and fixed during deployment (kept here so they don't recur)
- **`ModuleNotFoundError: No module named 'mlflow'`** — `model_pool.py` had
  a top-level `import mlflow`, but mlflow isn't in `requirements.txt` on
  purpose (dev-only, heavy dependency). Fixed by moving the import inside
  `if __name__ == "__main__":`.
- **`Cannot save file into a non-existent directory: 'data/monthly'`** —
  `pipeline_orchestrator.py`'s `aggregate_monthly()` assumed the folder
  already existed (true locally, false on a fresh Render deploy). Fixed
  with `os.makedirs(..., exist_ok=True)` before the write.
- **CORS blocked on Vercel preview URLs** — Vercel generates a new
  random-hash URL per deployment; an exact `ALLOWED_ORIGINS` match doesn't
  cover these. Fixed by adding `allow_origin_regex=r"https://.*\.vercel\.app"`.
- **Champion badge contradicted the visible scoreboard** — `champion_selection.py`
  originally ran its own separate 3-year evaluation, which could disagree
  with the averaged walk-forward numbers shown in the UI table. Fixed by
  rewriting it to derive the champion directly from `scoreboard_m7_folds.csv`
  — one source of truth, no possible contradiction.
- **Duplicate/misspelled file** — `Monte_carlo_forcast.py` (typo) and
  `monte_carlo_forecast.py` (correct) both existed at one point after a
  patch script created the correctly-named file without removing the old
  one. Consolidated to the single correctly-spelled file.

### Keep-alive
Render's free tier spins down after 15 minutes of no traffic, which can
kill an in-progress background job. cron-job.org pings `GET /health`
every 10 minutes to prevent this. `/health` is zero-dependency — no disk,
no pipeline logic.

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
- **200** — known pilot or previously-processed location: instant result.
- **202** — brand-new coordinate: background job started. Response includes
  `job_id` and `check_status_url`. Confirmed working end-to-end on Render's
  live infrastructure (tested against Chennai, a non-pilot location).

Response shape (both 200 and completed-job responses):
```json
{
  "champion_model": "ETS",
  "scoreboard": [ {"model": "ETS", "nrmse": 0.0848, "smape": 3.17, "r2": 0.8936}, ... ],
  "annual_forecast": [ {"year": 2027, "p50_kwh": ..., "p75_kwh": ..., "p90_kwh": ...}, ... ],
  "historical_monthly_avg": [ {"month": 1, "avg_kwh": ...}, ... ],       // 12 rows
  "historical_monthly_full": [ {"year": 2001, "month": 1, "kwh": ...}, ... ], // ~300 rows
  "summary": { "total_10yr_mwh": ..., "specific_yield_kwh_per_kwp": ..., ... }
}
```

### `GET /v1/forecast/status/{job_id}`
Poll until `"status": "completed"` or `"failed"`.

### `GET /health`
`{"status": "ok"}` — used by the keep-alive cron.

---

## Model accuracy — honest history

An earlier version scored R² of 0.97-0.99 by accidentally feeding models
real *future* weather as an input — a leak, since a genuine forecast never
has that. Fixed by restricting features to the target's own history only.
Honest R² dropped to ~0.82 average across the 3 pilots.

Two legitimate improvement attempts were made afterward:
- **Adding ETS** (a statistical model named in the spec's Section 8.1 but
  not originally built) — genuine improvement, now the champion at all 3
  pilot sites. Average R² up to ~0.84.
- **Pooling all 3 locations into one shared model** — made every site
  *worse*, not better. Kept as `pooled_model.py`, documented, not deployed.
  Confirms each site's weather pattern is distinct enough that per-location
  models (the current architecture) are the right call.

**0.82-0.89 (varies by site) is the honest, validated ceiling** with the
data currently available — not a target to force higher without
reintroducing leakage or overfitting. Getting materially past it requires
real metered generation data from an actual installed system, which no
client has provided yet.

---


## The 3 original pilot sites

| Location | ID | Coordinates |
|---|---|---|
| Jaisalmer, RJ | `IN_JSL_001` | 26.9157, 70.9083 |
| Kolkata, WB | `IN_KOL_001` | 22.5726, 88.3639 |
| Pune, MH | `IN_PUN_001` | 18.5204, 73.8567 |

Any other coordinate is supported via the async pipeline, confirmed
working in production (tested against Chennai, 13.0827/80.2707).
