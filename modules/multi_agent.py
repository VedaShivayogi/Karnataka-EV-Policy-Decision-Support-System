"""
Multi-Agent AI
Three specialist agents debate/analyze a proposed policy change, then a
coordinator agent merges their views into one recommendation.

Agents:
  1. PolicyAgent      - legal/regulatory feasibility
  2. BudgetAgent       - fiscal cost & ROI
  3. EnvironmentAgent  - emissions/environmental impact

Run:
    python modules/multi_agent.py "Increase EV purchase subsidy by ₹5,000 in Karnataka"
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.llm_client import chat
from modules.rag_policy_chat import retrieve


AGENTS = {
    "PolicyAgent": (
        "You are a government policy & regulatory analyst. Assess legal feasibility, "
        "implementation timeline, and administrative complexity of the proposal. "
        "Answer in 3 concise bullet points."
    ),
    "BudgetAgent": (
        "You are a public finance analyst. Estimate fiscal cost, funding sources, and "
        "expected ROI (e.g. tax revenue from EV sales, reduced fuel subsidy outlay) of "
        "the proposal. Answer in 3 concise bullet points with rough numeric estimates "
        "clearly labeled as estimates."
    ),
    "EnvironmentAgent": (
        "You are an environmental impact analyst. Assess likely CO2/emissions reduction "
        "and any downside environmental tradeoffs (e.g. battery disposal, grid load) of "
        "the proposal. Answer in 3 concise bullet points."
    ),
}

COORDINATOR_PROMPT = (
    "You are the lead policy coordinator. Given the three specialist analyses below, "
    "write a final recommendation: Approve / Approve with modifications / Reject, "
    "with a one-paragraph justification that weighs all three perspectives."
)


def run_agents(proposal: str):
    # ground agents in retrieved policy context where available
    try:
        chunks = retrieve(proposal, k=3)
        context = "\n".join(f"- {c['text']}" for c in chunks)
    except Exception:
        context = "(no policy documents indexed yet)"

    results = {}
    for name, system in AGENTS.items():
        prompt = f"Proposal: {proposal}\n\nRelevant policy context:\n{context}\n\nYour analysis:"
        results[name] = chat(prompt, system=system)

    combined = "\n\n".join(f"### {name}\n{text}" for name, text in results.items())
    final = chat(f"Proposal: {proposal}\n\n{combined}\n\nGive your final recommendation.",
                 system=COORDINATOR_PROMPT)
    results["Coordinator"] = final
    return results


if __name__ == "__main__":
    proposal = sys.argv[1] if len(sys.argv) > 1 else "Increase EV purchase subsidy by ₹5,000 in Karnataka"
    out = run_agents(proposal)
    for name, text in out.items():
        print(f"\n=== {name} ===\n{text}")
