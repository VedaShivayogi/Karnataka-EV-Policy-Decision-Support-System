"""
Policy Comparison
Compares Karnataka's EV policy with Tamil Nadu / Maharashtra using the RAG
index (if those state docs are present in data/policy_docs/) plus an LLM
summary table.

Run:
    python modules/policy_comparison.py --states Karnataka "Tamil Nadu" Maharashtra
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.rag_policy_chat import retrieve
from modules.llm_client import chat

SYSTEM_PROMPT = (
    "You are a comparative EV policy analyst. Using only the provided context snippets, "
    "produce a markdown comparison table with rows: Purchase Subsidy, Road Tax/Registration "
    "Waiver, Charging Infra Target, Manufacturing Incentives, Policy Duration. Columns are the "
    "requested states. Write 'Not specified in provided documents' for any missing cell — "
    "never invent numbers. After the table, add a 2-3 sentence takeaway."
)


def compare(states):
    context_blocks = []
    for state in states:
        chunks = retrieve(f"{state} EV policy subsidy charging infrastructure incentives", k=4)
        joined = "\n".join(f"- {c['text']}" for c in chunks)
        context_blocks.append(f"### {state}\n{joined}")

    context = "\n\n".join(context_blocks)
    prompt = f"States to compare: {', '.join(states)}\n\nContext:\n{context}\n\nBuild the comparison."
    return chat(prompt, system=SYSTEM_PROMPT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="+", default=["Karnataka", "Tamil Nadu", "Maharashtra"])
    args = parser.parse_args()
    print(compare(args.states))
