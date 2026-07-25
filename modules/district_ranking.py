"""
District Ranking
Ranks Karnataka districts by "EV readiness" using a weighted composite score
of infrastructure, income, urbanization, and existing adoption, then asks the
LLM to explain the ranking in plain language.

Run:
    python modules/district_ranking.py
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.llm_client import chat

WEIGHTS = {
    "charging_stations": 0.30,
    "avg_income_inr": 0.20,
    "urbanization_pct": 0.25,
    "ev_adoption_pct": 0.25,
}


def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


def rank_districts(path=os.path.join(config.DATA_DIR, "district_data.csv")):
    df = pd.read_csv(path)
    score = pd.Series(0.0, index=df.index)
    for col, w in WEIGHTS.items():
        score += normalize(df[col]) * w
    df["readiness_score"] = (score * 100).round(1)
    df = df.sort_values("readiness_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df[["rank", "district", "readiness_score"] + list(WEIGHTS.keys())]


def explain_ranking(ranked_df, top_n=5):
    table = ranked_df.head(top_n).to_string(index=False)
    prompt = (
        f"Here is a table ranking Karnataka districts by EV readiness score (0-100):\n\n{table}\n\n"
        "In 4-6 bullet points, explain what makes the top districts EV-ready and give one "
        "concrete policy recommendation for the lowest-ranked district in this table."
    )
    return chat(prompt, system="You are an EV infrastructure policy analyst.")


if __name__ == "__main__":
    ranked = rank_districts()
    print(ranked.to_string(index=False))
    print("\n--- LLM Explanation ---\n")
    print(explain_ranking(ranked))

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    ranked.to_csv(os.path.join(config.OUTPUT_DIR, "district_ranking.csv"), index=False)
