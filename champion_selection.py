"""
Milestone 8 -- Champion selection, now derived DIRECTLY from the same
walk-forward fold averages shown in the UI scoreboard (data/scoreboard_m7_folds.csv),
instead of running a separate third evaluation. This guarantees the
displayed table and the champion badge can never contradict each other --
whichever model has the lowest averaged nRMSE across both walk-forward
folds is the champion, full stop.
"""

import pandas as pd

FOLDS_PATH = "data/scoreboard_m7_folds.csv"
OUTPUT_PATH = "data/champion_decisions_m8.csv"


def select_champions():
    df = pd.read_csv(FOLDS_PATH)
    averaged = df.groupby(["location_id", "model"])[["nrmse", "smape", "r2"]].mean().reset_index()

    decisions = []
    for location_id, group in averaged.groupby("location_id"):
        champion_row = group.loc[group["nrmse"].idxmin()]
        decisions.append({
            "location_id": location_id,
            "champion_type": "single",
            "champion_model": champion_row["model"],
            "nrmse": champion_row["nrmse"],
            "smape": champion_row["smape"],
            "r2": champion_row["r2"],
        })
        print(f"{location_id}: champion = {champion_row['model']} "
              f"(avg nRMSE={champion_row['nrmse']:.4f}, R2={champion_row['r2']:.4f})")

    pd.DataFrame(decisions).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    select_champions()
