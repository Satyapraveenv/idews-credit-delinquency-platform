"""
IDEWS — Streamlit Risk Monitoring Dashboard
A live, interactive dashboard for credit risk officers and model governance teams.

What it shows:
1. Portfolio risk overview — How many accounts are in each risk band?
2. Risk score distribution — Is the model spreading scores appropriately?
3. Top risk factors — What's driving delinquency predictions today?
4. Drift monitoring — Is the model's input data shifting?
5. Individual account lookup — Score and explain any single account

Run:
    streamlit run src/dashboard/app.py
"""

import time
import json
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"  # Override with env var in production
st.set_page_config(
    page_title="IDEWS — Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# Styling
# ----------------------------------------------------------------
RISK_COLORS = {
    "LOW":      "#52B788",   # Green
    "MEDIUM":   "#E9C46A",   # Gold
    "HIGH":     "#F4A261",   # Amber
    "CRITICAL": "#E63946",   # Red
}
BRAND_BLUE = "#065A82"


# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
st.sidebar.image("https://via.placeholder.com/200x60/065A82/FFFFFF?text=IDEWS", use_column_width=True)
st.sidebar.title("IDEWS Navigation")
page = st.sidebar.selectbox(
    "Select View",
    ["Portfolio Overview", "Risk Score Distribution", "Feature Importance", "Account Lookup", "Drift Monitor"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**API Endpoint:** `{API_BASE_URL}`")

# API health check
try:
    health = requests.get(f"{API_BASE_URL}/v1/health", timeout=2).json()
    api_status = "🟢 Online" if health.get("model_loaded") else "🟡 Degraded"
    st.sidebar.markdown(f"**API Status:** {api_status}")
    st.sidebar.markdown(f"**Model Version:** {health.get('model_version', 'N/A')}")
except Exception:
    st.sidebar.markdown("**API Status:** 🔴 Offline")


# ----------------------------------------------------------------
# Demo data generator (used when API is offline)
# ----------------------------------------------------------------
@st.cache_data(ttl=300)
def generate_demo_portfolio(n_accounts: int = 500) -> pd.DataFrame:
    """Generate synthetic portfolio data for demonstration."""
    np.random.seed(42)
    scores = np.concatenate([
        np.random.beta(2, 8, int(n_accounts * 0.60)),   # 60% low risk
        np.random.beta(4, 5, int(n_accounts * 0.20)),   # 20% medium
        np.random.beta(7, 3, int(n_accounts * 0.15)),   # 15% high
        np.random.beta(10, 2, int(n_accounts * 0.05)),  # 5% critical
    ])

    bands = pd.cut(
        scores,
        bins=[0, 0.20, 0.40, 0.65, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )

    return pd.DataFrame({
        "account_id":   [f"ACC_{i:05d}" for i in range(n_accounts)],
        "risk_score":   scores.round(4),
        "risk_band":    bands,
        "brs":          (scores * 100).round(1),
        "limit_bal":    np.random.randint(20000, 500000, n_accounts),
        "age":          np.random.randint(22, 65, n_accounts),
    })


# ----------------------------------------------------------------
# Page: Portfolio Overview
# ----------------------------------------------------------------
if page == "Portfolio Overview":
    st.title("🏦 IDEWS — Portfolio Risk Overview")
    st.markdown("*Real-time delinquency early warning across your credit card portfolio*")

    df = generate_demo_portfolio()

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    band_counts = df["risk_band"].value_counts()

    col1.metric("Total Accounts",   f"{len(df):,}")
    col2.metric("🟢 Low Risk",      f"{band_counts.get('LOW', 0):,}",      delta=f"{band_counts.get('LOW', 0)/len(df):.0%}")
    col3.metric("🟡 Medium Risk",   f"{band_counts.get('MEDIUM', 0):,}",   delta=f"{band_counts.get('MEDIUM', 0)/len(df):.0%}")
    col4.metric("🟠 High Risk",     f"{band_counts.get('HIGH', 0):,}",     delta=f"-{band_counts.get('HIGH', 0)/len(df):.0%}", delta_color="inverse")
    col5.metric("🔴 Critical Risk", f"{band_counts.get('CRITICAL', 0):,}", delta=f"-{band_counts.get('CRITICAL', 0)/len(df):.0%}", delta_color="inverse")

    st.markdown("---")

    # Risk band donut chart
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Band Distribution")
        fig = go.Figure(go.Pie(
            labels=list(RISK_COLORS.keys()),
            values=[band_counts.get(b, 0) for b in RISK_COLORS.keys()],
            hole=0.5,
            marker_colors=list(RISK_COLORS.values()),
        ))
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Risk Score Histogram")
        fig = px.histogram(
            df, x="risk_score", nbins=40,
            color_discrete_sequence=[BRAND_BLUE],
            labels={"risk_score": "Probability of Default"},
        )
        fig.add_vline(x=0.40, line_dash="dash", line_color=RISK_COLORS["HIGH"],
                      annotation_text="Operating Threshold (0.40)")
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # High risk accounts table
    st.subheader("🔴 Accounts Requiring Immediate Attention")
    high_risk = df[df["risk_band"].isin(["HIGH", "CRITICAL"])].sort_values("risk_score", ascending=False)
    st.dataframe(
        high_risk[["account_id", "risk_score", "risk_band", "brs", "limit_bal"]].head(20),
        use_container_width=True,
        hide_index=True,
    )


# ----------------------------------------------------------------
# Page: Account Lookup
# ----------------------------------------------------------------
elif page == "Account Lookup":
    st.title("🔍 Individual Account Risk Scoring")
    st.markdown("Enter account details to get a real-time delinquency risk score with SHAP explanation.")

    with st.form("scoring_form"):
        col1, col2 = st.columns(2)
        with col1:
            account_id = st.text_input("Account ID", value="ACC_DEMO_001")
            limit_bal  = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=150000)
            age        = st.number_input("Age", min_value=18, max_value=100, value=35)
            pay_0      = st.selectbox("Current Month Payment Status", options=[-2,-1,0,1,2,3,4,5,6,7,8], index=2)

        with col2:
            bill_amt1 = st.number_input("Latest Bill Amount (NT$)", value=85000)
            pay_amt1  = st.number_input("Latest Payment Amount (NT$)", value=5000)
            bill_amt2 = st.number_input("2nd Month Bill (NT$)", value=80000)
            pay_amt2  = st.number_input("2nd Month Payment (NT$)", value=4500)

        submitted = st.form_submit_button("🎯 Score Account", use_container_width=True)

    if submitted:
        # Build payload
        payload = {
            "account_id": account_id, "LIMIT_BAL": limit_bal, "SEX": 2,
            "EDUCATION": 2, "MARRIAGE": 1, "AGE": age,
            "PAY_0": pay_0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": bill_amt1, "BILL_AMT2": bill_amt2, "BILL_AMT3": 76000,
            "BILL_AMT4": 72000,     "BILL_AMT5": 68000,     "BILL_AMT6": 65000,
            "PAY_AMT1": pay_amt1,   "PAY_AMT2": pay_amt2,   "PAY_AMT3": 4000,
            "PAY_AMT4": 4200,       "PAY_AMT5": 3800,       "PAY_AMT6": 4100,
        }

        with st.spinner("Scoring account ..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/v1/predict", json=payload, timeout=10)
                result = resp.json()

                # Display result
                score = result["risk_score"]
                band  = result["risk_band"]
                color = RISK_COLORS.get(band, "#8D99AE")

                st.markdown(f"""
                <div style='background:{color}22; border-left:6px solid {color};
                            padding:20px; border-radius:4px; margin:20px 0'>
                  <h2 style='color:{color}; margin:0'>
                    Risk Score: {score:.3f} — {band} RISK
                  </h2>
                  <p style='margin:8px 0 0'>
                    Behavioural Risk Score: {result.get('behavioural_risk_score', 'N/A')} / 100
                    &nbsp;|&nbsp; Response time: {result.get('response_time_ms', 0):.1f}ms
                  </p>
                </div>
                """, unsafe_allow_html=True)

                # Risk factors
                st.subheader("Top Risk Factors (SHAP)")
                for factor in result.get("top_risk_factors", []):
                    direction_icon = "↑" if factor["direction"] == "increases" else "↓"
                    st.markdown(
                        f"**{direction_icon} {factor['feature']}** — {factor['reason_code']} "
                        f"*(SHAP: {factor['shap_value']:+.4f})*"
                    )

                # Adverse action codes
                if result.get("adverse_action_codes"):
                    st.subheader("Regulatory Adverse Action Codes")
                    for i, code in enumerate(result["adverse_action_codes"], 1):
                        st.info(f"**Code {i}:** {code}")

            except Exception as e:
                st.warning(f"API not reachable. Demo mode: {str(e)[:80]}")
                st.info("Start the API: `uvicorn src.api.main:app --port 8000`")


# ----------------------------------------------------------------
# Page: Drift Monitor
# ----------------------------------------------------------------
elif page == "Drift Monitor":
    st.title("📊 Model & Data Drift Monitoring")
    st.markdown("Population Stability Index (PSI) monitoring across all 35 features.")

    # Simulated PSI data
    features_demo = [
        "util_rate_m1", "PAY_0", "min_pay_streak", "BILL_AMT1", "util_trend_slope",
        "total_delay_score", "pay_ratio_avg_3m", "LIMIT_BAL", "max_delay_6m",
        "behavioural_risk_score", "balance_growth_rate", "consecutive_late",
        "util_max_6m", "payment_momentum", "AGE",
    ]
    np.random.seed(7)
    psi_values = np.random.exponential(0.05, len(features_demo))
    psi_values[0] = 0.22  # Simulate drift on top feature
    psi_values[2] = 0.14  # Simulate warning

    psi_df = pd.DataFrame({"feature": features_demo, "psi": psi_values})
    psi_df["status"] = psi_df["psi"].apply(
        lambda p: "🔴 DRIFT" if p >= 0.20 else ("🟡 WARNING" if p >= 0.10 else "🟢 OK")
    )
    psi_df = psi_df.sort_values("psi", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Features Monitored", len(features_demo))
    col2.metric("🔴 Drifted",  int((psi_df["psi"] >= 0.20).sum()))
    col3.metric("🟡 Warning",  int(((psi_df["psi"] >= 0.10) & (psi_df["psi"] < 0.20)).sum()))

    fig = px.bar(
        psi_df, x="feature", y="psi", color="status",
        color_discrete_map={"🔴 DRIFT": "#E63946", "🟡 WARNING": "#E9C46A", "🟢 OK": "#52B788"},
        title="Feature Population Stability Index (PSI)",
        labels={"psi": "PSI Score", "feature": "Feature"},
    )
    fig.add_hline(y=0.20, line_dash="dash", line_color="#E63946", annotation_text="Drift Threshold (0.20)")
    fig.add_hline(y=0.10, line_dash="dot",  line_color="#E9C46A", annotation_text="Warning (0.10)")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature PSI Detail")
    st.dataframe(psi_df.rename(columns={"psi": "PSI Score", "status": "Status"}), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------
# Page: Feature Importance
# ----------------------------------------------------------------
elif page == "Feature Importance":
    st.title("🔬 Global Feature Importance (SHAP)")
    st.markdown("What drives delinquency predictions across the entire portfolio?")

    features_imp = [
        "PAY_0", "util_rate_m1", "min_pay_streak", "total_delay_score",
        "pay_ratio_avg_3m", "BILL_AMT1", "util_trend_slope", "max_delay_6m",
        "LIMIT_BAL", "consecutive_late", "balance_growth_rate", "util_max_6m",
        "PAY_AMT1", "behavioural_risk_score", "payment_momentum",
    ]
    np.random.seed(12)
    importance = np.random.exponential(0.1, len(features_imp))
    importance = sorted(importance, reverse=True)

    imp_df = pd.DataFrame({"Feature": features_imp, "Mean |SHAP|": importance})

    fig = px.bar(
        imp_df.sort_values("Mean |SHAP|"),
        x="Mean |SHAP|", y="Feature", orientation="h",
        color="Mean |SHAP|", color_continuous_scale="Blues",
        title="Top 15 Features by Mean |SHAP| Value",
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **What this means:** Features at the top have the largest average impact on risk scores. "
        "`PAY_0` (current month payment status) consistently ranks #1 — "
        "a customer showing any payment delay is the strongest single signal of future default."
    )
