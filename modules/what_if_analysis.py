"""
What-If Analysis
Lets you save multiple policy scenarios (e.g. "Baseline", "+5000 subsidy",
"+10000 subsidy + 200 new stations") and compare predicted EV adoption
side by side, using the trained ML model from ml_forecast.py.

Run:
    python modules/what_if_analysis.py
"""

import sys
import os
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.ml_forecast import load_data, train_model, predict_scenario


def save_scenario(name, base_row, changes, prediction):
    os.makedirs(config.SCENARIO_DIR, exist_ok=True)
    path = os.path.join(config.SCENARIO_DIR, f"{name.replace(' ', '_')}.json")
    with open(path, "w") as f:
        json.dump(
            {"name": name, "base_row": base_row, "changes": changes, "predicted_adoption_pct": prediction},
            f,
            indent=2,
        )
    return path


def load_all_scenarios():
    if not os.path.isdir(config.SCENARIO_DIR):
        return []
    scenarios = []
    for fname in sorted(os.listdir(config.SCENARIO_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(config.SCENARIO_DIR, fname)) as f:
                scenarios.append(json.load(f))
    return scenarios


def compare_scenarios():
    scenarios = load_all_scenarios()
    if not scenarios:
        print("No saved scenarios yet. Run build_default_scenarios() first.")
        return None
    df = pd.DataFrame(
        [
            {"scenario": s["name"], "changes": s["changes"], "predicted_ev_adoption_pct": s["predicted_adoption_pct"]}
            for s in scenarios
        ]
    )
    return df.sort_values("predicted_ev_adoption_pct", ascending=False)


def build_default_scenarios(district_index=0):
    df = load_data()
    model = train_model(df)
    base_row = df.iloc[district_index].to_dict()

    scenario_defs = {
        "Baseline": {},
        "Subsidy +5000": {"subsidy_inr": base_row["subsidy_inr"] + 5000},
        "Subsidy +10000": {"subsidy_inr": base_row["subsidy_inr"] + 10000},
        "Subsidy +5000 & +200 stations": {
            "subsidy_inr": base_row["subsidy_inr"] + 5000,
            "charging_stations": base_row["charging_stations"] + 200,
        },
    }

    for name, changes in scenario_defs.items():
        pred = predict_scenario(model, base_row, changes)
        path = save_scenario(name, base_row, changes, pred)
        print(f"Saved '{name}' -> {pred:.2f}% predicted adoption ({path})")


if __name__ == "__main__":
    build_default_scenarios()
    print("\n--- Comparison ---")
    print(compare_scenarios().to_string(index=False))
