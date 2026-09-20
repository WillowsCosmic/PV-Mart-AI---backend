"""
Milestone 6 / Stage 6 (partial) — Dynamic Multi-Model Selection Layer,
first pass: candidate pool trained against the PHYSICS BASELINE series.


Candidate pool (matched to spec 8.2: 300 monthly rows favors
statistical/ML models over deep nets):
  - Persistence (naive: same month last year)
  - Linear Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - SARIMA (statsmodels)

Metrics (spec Section 12): R2, MAE, RMSE, nRMSE, sMAPE

"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

from site_configs import PILOT_SITES

warnings.filterwarnings("ignore")  

MONTHLY_DIR = "data/monthly"
TARGET_COLUMN = "PV_energy_kWh"  

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "solar-forecast-candidate-pool"

TEST_YEARS = 3


def load_monthly(location_id: str) -> pd.DataFrame:
    df = pd.read_parquet(f"{MONTHLY_DIR}/{location_id}_monthly.parquet")
    return df.sort_values(["year", "month"]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Only uses the target's OWN history + calendar position -- no raw
    weather columns (GHI/POA/temperature) as inputs. Those are near-
    deterministic transforms of PV_energy_kWh itself (Stage 5's formula),
    and more importantly, a real 10-year-ahead forecast would never have
    genuine future weather values to feed in. This keeps the model
    honestly limited to what a real forecast run would actually have
    access to."""
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["lag_1"] = df[TARGET_COLUMN].shift(1)
    df["lag_12"] = df[TARGET_COLUMN].shift(12)  
    df["rolling_mean_12"] = df[TARGET_COLUMN].shift(1).rolling(window=12).mean()
    return df.dropna().reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_year = df["year"].max() - test_years + 1
    train = df[df["year"] < cutoff_year]
    test = df[df["year"] >= cutoff_year]
    return train, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nrmse = rmse / (y_true.max() - y_true.min())
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    return {"r2": r2, "mae": mae, "rmse": rmse, "nrmse": nrmse, "smape": smape}


FEATURE_COLUMNS = ["month_sin", "month_cos", "lag_1", "lag_12", "rolling_mean_12"]


def train_persistence(train, test):
    """Naive baseline: predict the same month's value from last year."""
    return test["lag_12"].values


def train_linear_regression(train, test):
    model = LinearRegression()
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model.predict(test[FEATURE_COLUMNS])


def train_random_forest(train, test):
    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model.predict(test[FEATURE_COLUMNS])


def train_xgboost(train, test):
    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model.predict(test[FEATURE_COLUMNS])


def train_lightgbm(train, test):
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                random_state=42, verbose=-1)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model.predict(test[FEATURE_COLUMNS])


def train_sarima(train, test):
    series = train.set_index(pd.PeriodIndex(
        train["year"].astype(str) + "-" + train["month"].astype(str), freq="M"
    ))[TARGET_COLUMN]
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=len(test))
    return forecast.values


CANDIDATES = {
    "Persistence": train_persistence,
    "LinearRegression": train_linear_regression,
    "RandomForest": train_random_forest,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
    "SARIMA": train_sarima,
}


def run_candidate_pool_for_location(site: dict) -> pd.DataFrame:
    location_id = site["location_id"]
    print(f"\n=== {location_id} ({site['city']}) ===")

    df = load_monthly(location_id)
    df = add_lag_features(df)
    train, test = chronological_split(df, TEST_YEARS)

    print(f"  train: {train['year'].min()}-{train['year'].max()} "
          f"({len(train)} rows) | test: {test['year'].min()}-{test['year'].max()} "
          f"({len(test)} rows)")

    scoreboard_rows = []
    y_true = test[TARGET_COLUMN].values

    for model_name, train_fn in CANDIDATES.items():
        try:
            with mlflow.start_run(run_name=f"{location_id}_{model_name}", nested=True):
                y_pred = train_fn(train, test)
                metrics = compute_metrics(y_true, y_pred)

                mlflow.log_param("location_id", location_id)
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("train_years", f"{train['year'].min()}-{train['year'].max()}")
                mlflow.log_param("test_years", f"{test['year'].min()}-{test['year'].max()}")
                mlflow.log_metrics(metrics)

                scoreboard_rows.append({"model": model_name, "location_id": location_id, **metrics})
                print(f"  {model_name:18s} nRMSE={metrics['nrmse']:.4f}  "
                      f"sMAPE={metrics['smape']:.2f}%  R2={metrics['r2']:.4f}")
        except Exception as exc:
            print(f"  {model_name:18s} FAILED: {exc}")

    return pd.DataFrame(scoreboard_rows)


if __name__ == "__main__":
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    all_scoreboards = []
    for site in PILOT_SITES:
        with mlflow.start_run(run_name=f"candidate_pool_{site['location_id']}"):
            scoreboard = run_candidate_pool_for_location(site)
            all_scoreboards.append(scoreboard)

    full_scoreboard = pd.concat(all_scoreboards, ignore_index=True)
    full_scoreboard.to_csv("data/scoreboard_m6.csv", index=False)

    print("\n=== Champion per location (lowest nRMSE) ===")
    for location_id, group in full_scoreboard.groupby("location_id"):
        champion = group.loc[group["nrmse"].idxmin()]
        print(f"  {location_id}: {champion['model']} "
              f"(nRMSE={champion['nrmse']:.4f}, sMAPE={champion['smape']:.2f}%)")

    print("\nMilestone 6 (candidate pool, physics-series forecasting) complete.")
    print("Full scoreboard saved to data/scoreboard_m6.csv")