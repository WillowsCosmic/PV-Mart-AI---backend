"""
Milestone 8 / Stage 6.3 — Champion-Challenger Selection Logic

Implements Section 8.3's pseudocode as a reusable function:

    for each location:
        for each candidate model in POOL:
            train on chronological training window
            evaluate on chronological validation window
            record R2, MAE, RMSE, nRMSE, sMAPE
        champion = argmin(validation_error)   # never argmax(train_accuracy)
        if top_2_or_3_models are within noise band of champion:
            champion = weighted_ensemble(top_models, weights=inverse_error)
        store champion + full scoreboard for the location
    re-run this selection whenever new monthly data lands

This wraps Milestone 6/7's candidate pool + walk-forward machinery into a
single callable that always returns a champion decision (or ensemble)
based on validation error -- ready to be re-invoked on a schedule
(Airflow/Prefect, Milestone 11) whenever new monthly data arrives.

Run:
    python champion_selection.py
"""

import numpy as np
import pandas as pd
import mlflow

from site_configs import PILOT_SITES
from model_pool import (
    add_lag_features, compute_metrics, CANDIDATES, TARGET_COLUMN,
    MLFLOW_TRACKING_URI, load_monthly, chronological_split,
)

MLFLOW_EXPERIMENT = "solar-forecast-champion-selection"

# "Within noise band" threshold (Section 8.3): if a challenger's nRMSE is
# within this fraction of the champion's, treat it as tied and ensemble
# instead of picking one arbitrarily.
NOISE_BAND_PCT = 0.05  # 5%

VALIDATION_TEST_YEARS = 3


def select_champion(location_id: str, train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """Trains every candidate, scores on the validation window, and
    returns the champion decision -- a single model, or a weighted
    ensemble if multiple models are within the noise band of each other."""
    y_true = val[TARGET_COLUMN].values
    scoreboard = []
    predictions = {}

    for model_name, train_fn in CANDIDATES.items():
        try:
            y_pred = train_fn(train, val)
            metrics = compute_metrics(y_true, y_pred)
            scoreboard.append({"model": model_name, **metrics})
            predictions[model_name] = y_pred
        except Exception as exc:
            print(f"    {model_name} failed during selection: {exc}")

    scoreboard_df = pd.DataFrame(scoreboard).sort_values("nrmse").reset_index(drop=True)
    best_nrmse = scoreboard_df.iloc[0]["nrmse"]

    # Which models are within the noise band of the best one?
    within_band = scoreboard_df[
        scoreboard_df["nrmse"] <= best_nrmse * (1 + NOISE_BAND_PCT)
    ]

    if len(within_band) > 1:
        # Weighted ensemble: weight = inverse of validation error
        weights = 1.0 / within_band["nrmse"].values
        weights = weights / weights.sum()
        ensemble_pred = np.zeros_like(y_true, dtype=float)
        for w, model_name in zip(weights, within_band["model"]):
            ensemble_pred += w * predictions[model_name]
        ensemble_metrics = compute_metrics(y_true, ensemble_pred)

        champion_decision = {
            "location_id": location_id,
            "champion_type": "ensemble",
            "champion_model": " + ".join(within_band["model"].tolist()),
            "ensemble_weights": dict(zip(within_band["model"], weights)),
            **ensemble_metrics,
        }
    else:
        top = scoreboard_df.iloc[0]
        champion_decision = {
            "location_id": location_id,
            "champion_type": "single",
            "champion_model": top["model"],
            "ensemble_weights": None,
            "r2": top["r2"], "mae": top["mae"], "rmse": top["rmse"],
            "nrmse": top["nrmse"], "smape": top["smape"],
        }

    return champion_decision, scoreboard_df


def run_selection_for_all_locations() -> pd.DataFrame:
    """This is the function that Airflow/Prefect (Milestone 11) would
    call on a schedule -- re-runs selection from scratch each time,
    always trusting current validation error, never a cached decision."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    decisions = []

    for site in PILOT_SITES:
        location_id = site["location_id"]
        print(f"\n=== {location_id} ({site['city']}) — champion selection ===")

        df = load_monthly(location_id)
        df = add_lag_features(df)
        train, val = chronological_split(df, VALIDATION_TEST_YEARS)

        with mlflow.start_run(run_name=f"selection_{location_id}"):
            decision, scoreboard_df = select_champion(location_id, train, val)

            mlflow.log_param("location_id", location_id)
            mlflow.log_param("champion_type", decision["champion_type"])
            mlflow.log_param("champion_model", decision["champion_model"])
            mlflow.log_metrics({
                "nrmse": decision["nrmse"], "smape": decision["smape"],
                "r2": decision["r2"],
            })
            mlflow.log_dict(scoreboard_df.to_dict(), "full_scoreboard.json")

            print(f"  Full scoreboard (sorted by nRMSE):")
            print(scoreboard_df[["model", "nrmse", "smape", "r2"]].to_string(index=False))
            print(f"\n  DECISION: {decision['champion_type']} -> {decision['champion_model']}")
            if decision["ensemble_weights"]:
                for m, w in decision["ensemble_weights"].items():
                    print(f"    weight[{m}] = {w:.3f}")
            print(f"  nRMSE={decision['nrmse']:.4f}  sMAPE={decision['smape']:.2f}%")

        decisions.append(decision)

    return pd.DataFrame(decisions)


if __name__ == "__main__":
    decisions_df = run_selection_for_all_locations()
    decisions_df.drop(columns=["ensemble_weights"]).to_csv(
        "data/champion_decisions_m8.csv", index=False
    )

    print("\n=== Final champion decisions, all locations ===")
    print(decisions_df[["location_id", "champion_type", "champion_model", "nrmse", "smape"]]
          .to_string(index=False))

    print("\nMilestone 8 (champion-challenger auto-selection) complete.")
    print("This function (run_selection_for_all_locations) is what gets")
    print("scheduled to re-run whenever new monthly data lands (Milestone 11).")