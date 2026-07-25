"""
Streamlit Dashboard - Karnataka EV Policy Decision Support System
Ties together all modules into one interactive UI.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import news_summarizer, rag_policy_chat, ml_forecast, district_ranking, \
    policy_comparison, multi_agent, what_if_analysis, pdf_report

st.set_page_config(page_title="Karnataka EV Policy Dashboard", layout="wide")
st.title("🔋 Karnataka EV Policy Decision Support System")

tabs = st.tabs([
    "📰 News Summarizer", "💬 Policy Chat", "📊 ML Forecast", "🏙️ District Ranking",
    "⚖️ Policy Comparison", "🤖 Multi-Agent Analysis", "🔁 What-If Scenarios", "📄 PDF Report"
])

with tabs[0]:
    st.subheader("Recent EV News Summary")
    if st.button("Fetch & Summarize Latest News"):
        with st.spinner("Fetching and summarizing..."):
            summary = news_summarizer.run()
        st.markdown(summary)

with tabs[1]:
    st.subheader("Ask about Karnataka's EV policy")
    q = st.text_input("Question", "What happens if Karnataka increases subsidies by ₹5,000?")
    if st.button("Ask"):
        with st.spinner("Retrieving & answering..."):
            answer, sources = rag_policy_chat.ask(q)
        st.write(answer)
        with st.expander("Sources used"):
            for s in sources:
                st.write(f"**{s['source']}** (score={s['score']:.3f})")
                st.caption(s["text"][:300] + "...")

with tabs[2]:
    st.subheader("EV Adoption Forecast")
    df = ml_forecast.load_data()
    st.dataframe(df)
    if st.button("Train model & forecast"):
        model = ml_forecast.train_model(df)
        sample = df.iloc[0].to_dict()
        baseline = ml_forecast.predict_scenario(model, sample)
        st.metric(f"Predicted EV adoption — {sample.get('district','district')}", f"{baseline:.2f}%")

with tabs[3]:
    st.subheader("District EV Readiness Ranking")
    ranked = district_ranking.rank_districts()
    st.dataframe(ranked, use_container_width=True)
    if st.button("Explain ranking with AI"):
        with st.spinner("Analyzing..."):
            st.write(district_ranking.explain_ranking(ranked))

with tabs[4]:
    st.subheader("Compare State EV Policies")
    states = st.multiselect("States", ["Karnataka", "Tamil Nadu", "Maharashtra", "Delhi"],
                             default=["Karnataka", "Tamil Nadu", "Maharashtra"])
    if st.button("Compare"):
        with st.spinner("Comparing..."):
            st.markdown(policy_comparison.compare(states))

with tabs[5]:
    st.subheader("Multi-Agent Policy Analysis")
    proposal = st.text_area("Proposal", "Increase EV purchase subsidy by ₹5,000 in Karnataka")
    if st.button("Run agents"):
        with st.spinner("Agents analyzing..."):
            results = multi_agent.run_agents(proposal)
        for name, text in results.items():
            st.markdown(f"**{name}**")
            st.write(text)

with tabs[6]:
    st.subheader("What-If Scenario Comparison")
    if st.button("Build default scenarios"):
        with st.spinner("Training model & running scenarios..."):
            what_if_analysis.build_default_scenarios()
    cmp = what_if_analysis.compare_scenarios()
    if cmp is not None:
        st.dataframe(cmp, use_container_width=True)
        st.bar_chart(cmp.set_index("scenario")["predicted_ev_adoption_pct"])

with tabs[7]:
    st.subheader("Generate Automatic PDF Report")
    if st.button("Generate Report"):
        with st.spinner("Building report..."):
            path = pdf_report.build_report()
        with open(path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="EV_Policy_Report.pdf")
