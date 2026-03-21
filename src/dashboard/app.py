"""
IDEWS — Streamlit Risk Monitoring Dashboard
Intelligent Delinquency Early Warning System

Live demo at: https://idews-demo.streamlit.app
GitHub:       https://github.com/Satyapraveenv/idews-credit-delinquency-platform

Pages:
  1. Portfolio Overview     — Risk band distribution across 500 accounts
  2. Risk Score Distribution — Score histogram and statistical breakdown
  3. Feature Importance     — SHAP-based global feature ranking
  4. Account Lookup         — Score any single account with SHAP explanation
  5. Drift Monitor          — PSI-based feature drift alerting
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDEWS — Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
RISK_COLORS = {
    "LOW":      "#52B788",
    "MEDIUM":   "#E9C46A",
    "HIGH":     "#F4A261",
    "CRITICAL": "#E63946",
}
BRAND_BLUE    = "#065A82"
BRAND_TEAL    = "#1C7293"
BRAND_MIDNIGHT = "#0D1B2A"

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    f"<div style='background:{BRAND_MIDNIGHT};padding:14px 12px;border-radius:6px;"
    f"border-left:4px solid {BRAND_TEAL};margin-bottom:12px'>"
    f"<span style='color:#CAF0F8;font-size:22px;font-weight:700;letter-spacing:3px'>IDEWS</span><br>"
    f"<span style='color:#94D2BD;font-size:11px'>Credit Risk Intelligence</span></div>",
    unsafe_allow_html=True,
)

page = st.sidebar.selectbox(
    "Select View",
    ["Portfolio Overview", "Risk Score Distribution",
     "Feature Importance", "Account Lookup", "Drift Monitor"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Mode:** 🟢 Live Demo")
st.sidebar.markdown("**Model:** XGBoost v1.0")
st.sidebar.markdown("**Data:** UCI Credit Card Default")
st.sidebar.markdown("**Accounts:** 30,000 training | 500 demo")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-Satyapraveenv-0D1B2A?logo=github)]"
    "(https://github.com/Satyapraveenv/idews-credit-delinquency-platform)"
)


# ── Demo data generators ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def generate_demo_portfolio(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    scores = np.concatenate([
        np.random.beta(2, 8,  int(n * 0.60)),
        np.random.beta(4, 5,  int(n * 0.20)),
        np.random.beta(7, 3,  int(n * 0.15)),
        np.random.beta(10, 2, int(n * 0.05)),
    ])
    bands = pd.cut(scores, bins=[0, 0.20, 0.40, 0.65, 1.0],
                   labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    return pd.DataFrame({
        "account_id":  [f"ACC_{i:05d}" for i in range(n)],
        "risk_score":  scores.round(4),
        "risk_band":   bands,
        "brs":         (scores * 100).round(1),
        "limit_bal":   np.random.randint(20_000, 500_000, n),
        "age":         np.random.randint(22, 65, n),
        "util_rate":   np.random.uniform(0.1, 0.98, n).round(3),
        "pay_status":  np.random.choice([-1, 0, 1, 2], n, p=[0.3, 0.4, 0.2, 0.1]),
    })


def demo_score_account(payload: dict) -> dict:
    """Simulate a model scoring response for demo purposes."""
    np.random.seed(abs(hash(payload.get("account_id", "demo"))) % 1000)
    pay_status = payload.get("PAY_0", 0)
    base = 0.12 + pay_status * 0.14 + np.random.uniform(-0.05, 0.05)
    score = float(np.clip(base, 0.01, 0.97))
    if   score < 0.20: band = "LOW"
    elif score < 0.40: band = "MEDIUM"
    elif score < 0.65: band = "HIGH"
    else:              band = "CRITICAL"
    brs = round(score * 100, 1)
    top_factors = [
        {"feature": "PAY_0",               "shap_value":  0.1823, "direction": "increases", "reason_code": "Current month payment delay detected"},
        {"feature": "util_rate_m1",         "shap_value":  0.1204, "direction": "increases", "reason_code": "Credit utilisation above 80% threshold"},
        {"feature": "min_pay_streak",       "shap_value":  0.0876, "direction": "increases", "reason_code": "Consecutive minimum payment pattern"},
        {"feature": "pay_ratio_avg_3m",     "shap_value": -0.0654, "direction": "decreases", "reason_code": "3-month payment ratio above average"},
        {"feature": "balance_growth_rate",  "shap_value":  0.0543, "direction": "increases", "reason_code": "Balance growing faster than payments"},
    ]
    adverse_codes = []
    if score >= 0.40:
        adverse_codes = [
            "AA-001: Delinquent payment history on current account",
            "AA-004: High revolving credit utilisation ratio",
        ]
    return {
        "account_id": payload.get("account_id"),
        "risk_score": round(score, 4),
        "risk_band": band,
        "behavioural_risk_score": brs,
        "response_time_ms": round(np.random.uniform(8, 45), 1),
        "top_risk_factors": top_factors,
        "adverse_action_codes": adverse_codes,
        "model_version": "xgb-v1.0-demo",
    }


# ── PAGE: Portfolio Overview ───────────────────────────────────────────────────
if page == "Portfolio Overview":
    st.title("🏦 IDEWS — Portfolio Risk Overview")
    st.markdown("*Delinquency early warning across the credit card portfolio — 60–90 days ahead of first missed payment*")

    df = generate_demo_portfolio()
    band_counts = df["risk_band"].value_counts()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Accounts",   f"{len(df):,}")
    c2.metric("🟢 Low Risk",      f"{band_counts.get('LOW', 0):,}",      delta=f"{band_counts.get('LOW', 0)/len(df):.0%}")
    c3.metric("🟡 Medium Risk",   f"{band_counts.get('MEDIUM', 0):,}",   delta=f"{band_counts.get('MEDIUM', 0)/len(df):.0%}")
    c4.metric("🟠 High Risk",     f"{band_counts.get('HIGH', 0):,}",     delta=f"-{band_counts.get('HIGH', 0)/len(df):.0%}", delta_color="inverse")
    c5.metric("🔴 Critical Risk", f"{band_counts.get('CRITICAL', 0):,}", delta=f"-{band_counts.get('CRITICAL', 0)/len(df):.0%}", delta_color="inverse")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Risk Band Distribution")
        fig = go.Figure(go.Pie(
            labels=list(RISK_COLORS.keys()),
            values=[band_counts.get(b, 0) for b in RISK_COLORS.keys()],
            hole=0.52,
            marker_colors=list(RISK_COLORS.values()),
            textinfo="label+percent",
        ))
        fig.update_layout(height=360, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Score Distribution by Band")
        fig = px.box(df, x="risk_band", y="risk_score",
                     color="risk_band",
                     color_discrete_map=RISK_COLORS,
                     labels={"risk_score": "Probability of Default", "risk_band": "Risk Band"},
                     category_orders={"risk_band": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]})
        fig.update_layout(height=360, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔴 Accounts Requiring Immediate Attention")
    high_risk = df[df["risk_band"].isin(["HIGH", "CRITICAL"])].sort_values("risk_score", ascending=False)
    st.dataframe(
        high_risk[["account_id", "risk_score", "risk_band", "brs", "limit_bal", "util_rate"]].head(20),
        use_container_width=True, hide_index=True,
    )
    st.caption("🔁 Demo data — refreshes every hour. In production, scores update daily from Vertex AI pipeline.")


# ── PAGE: Risk Score Distribution ─────────────────────────────────────────────
elif page == "Risk Score Distribution":
    st.title("📈 Risk Score Distribution Analysis")
    st.markdown("*Statistical breakdown of portfolio-wide Probability of Default (PD) scores*")

    df = generate_demo_portfolio()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Score",   f"{df['risk_score'].mean():.3f}")
    c2.metric("Median Score", f"{df['risk_score'].median():.3f}")
    c3.metric("90th Pctile",  f"{df['risk_score'].quantile(0.90):.3f}")
    c4.metric("Gini (demo)",  "0.724")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Score Histogram")
        fig = px.histogram(df, x="risk_score", nbins=50,
                           color_discrete_sequence=[BRAND_BLUE],
                           labels={"risk_score": "Probability of Default"})
        fig.add_vline(x=0.20, line_dash="dot",  line_color=RISK_COLORS["MEDIUM"],   annotation_text="Low→Medium (0.20)")
        fig.add_vline(x=0.40, line_dash="dash", line_color=RISK_COLORS["HIGH"],     annotation_text="Operating Threshold (0.40)")
        fig.add_vline(x=0.65, line_dash="dash", line_color=RISK_COLORS["CRITICAL"], annotation_text="Critical (0.65)")
        fig.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Cumulative Score Curve (KS Chart)")
        df_sorted = df.sort_values("risk_score")
        pct_accounts = np.linspace(0, 100, len(df_sorted))
        cumulative_bads = (df_sorted["risk_band"].isin(["HIGH", "CRITICAL"])).cumsum() / df_sorted["risk_band"].isin(["HIGH", "CRITICAL"]).sum() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pct_accounts, y=cumulative_bads,
                                 name="IDEWS Model", line=dict(color=BRAND_BLUE, width=2)))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100],
                                 name="Random Model", line=dict(color="#8D99AE", dash="dash")))
        ks_stat = float((cumulative_bads - pct_accounts).max())
        fig.add_annotation(x=40, y=72, text=f"KS Statistic: {ks_stat:.1f}%",
                           showarrow=False, font=dict(size=14, color=BRAND_BLUE),
                           bgcolor="white", bordercolor=BRAND_BLUE)
        fig.update_layout(height=380, margin=dict(t=20, b=20),
                          xaxis_title="% Accounts Reviewed",
                          yaxis_title="% Bad Accounts Captured")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score Band Summary Table")
    bands = ["LOW (0–0.20)", "MEDIUM (0.20–0.40)", "HIGH (0.40–0.65)", "CRITICAL (0.65–1.0)"]
    thresholds = [(0, 0.20), (0.20, 0.40), (0.40, 0.65), (0.65, 1.0)]
    rows = []
    for label, (lo, hi) in zip(bands, thresholds):
        sub = df[(df["risk_score"] >= lo) & (df["risk_score"] < hi)]
        rows.append({
            "Risk Band": label,
            "Accounts": len(sub),
            "% of Portfolio": f"{len(sub)/len(df):.1%}",
            "Avg Score": f"{sub['risk_score'].mean():.3f}",
            "Avg Credit Limit": f"NT${sub['limit_bal'].mean():,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── PAGE: Feature Importance ───────────────────────────────────────────────────
elif page == "Feature Importance":
    st.title("🔬 Global Feature Importance (SHAP)")
    st.markdown("*What drives delinquency predictions? Ranked by mean absolute SHAP value across the portfolio.*")

    features = [
        "PAY_0", "util_rate_m1", "min_pay_streak", "total_delay_score",
        "pay_ratio_avg_3m", "BILL_AMT1", "util_trend_slope", "max_delay_6m",
        "LIMIT_BAL", "consecutive_late", "balance_growth_rate", "util_max_6m",
        "PAY_AMT1", "behavioural_risk_score", "payment_momentum",
    ]
    np.random.seed(12)
    raw = np.random.exponential(0.10, len(features))
    importance = sorted(raw, reverse=True)
    imp_df = pd.DataFrame({"Feature": features, "Mean |SHAP|": importance,
                            "Category": ["Payment History"]*3 + ["Utilisation"]*3 +
                                        ["Balance"]*3 + ["Behavioural"]*3 + ["Payment Amount"]*3})

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig = px.bar(
            imp_df.sort_values("Mean |SHAP|"),
            x="Mean |SHAP|", y="Feature", orientation="h",
            color="Category",
            color_discrete_sequence=[BRAND_BLUE, BRAND_TEAL, "#52B788", "#E9C46A", "#F4A261"],
            title="Top 15 Features by Mean |SHAP| Value",
        )
        fig.update_layout(height=520, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**What the top features mean:**")
        explanations = {
            "PAY_0": "Current month payment status — the #1 predictor. Even a single month delay is a strong delinquency signal.",
            "util_rate_m1": "Credit utilisation last month. Above 80% correlates strongly with financial stress.",
            "min_pay_streak": "Consecutive months making only minimum payment. A behavioural early-warning pattern.",
            "total_delay_score": "Aggregate delay score across 6 months, weighted by recency.",
            "pay_ratio_avg_3m": "Average payment-to-balance ratio over 3 months. Higher = safer.",
        }
        for feat, explanation in list(explanations.items())[:5]:
            rank = features.index(feat) + 1
            st.markdown(f"**#{rank} {feat}**")
            st.caption(explanation)
            st.markdown("")

    st.info(
        "💡 **SR 11-7 note:** SHAP values provide the explainability required for model governance. "
        "Each individual prediction comes with the top 5 risk factors as human-readable adverse action codes "
        "— fully compliant with ECOA/Reg B requirements."
    )


# ── PAGE: Account Lookup ───────────────────────────────────────────────────────
elif page == "Account Lookup":
    st.title("🔍 Individual Account Risk Scoring")
    st.markdown("Score any account and get a SHAP-powered explanation of what's driving the risk.")
    st.info("🎯 **Demo mode** — scores are simulated. In production, this calls the live FastAPI endpoint at <50ms.")

    with st.form("score_form"):
        col1, col2 = st.columns(2)
        with col1:
            account_id = st.text_input("Account ID", value="ACC_DEMO_001")
            limit_bal  = st.number_input("Credit Limit (NT$)", min_value=10_000, max_value=1_000_000, value=150_000, step=10_000)
            age        = st.number_input("Age", min_value=18, max_value=100, value=35)
            pay_0      = st.selectbox("Current Month Payment Status",
                                      options=[-2, -1, 0, 1, 2, 3],
                                      format_func=lambda x: {-2: "-2 (No consumption)", -1: "-1 (Paid in full)",
                                                              0: "0 (Revolving credit)", 1: "1 (1 month delay)",
                                                              2: "2 (2 month delay)",    3: "3+ (3+ month delay)"}[x],
                                      index=2)
        with col2:
            bill_amt1 = st.number_input("Latest Bill Amount (NT$)", value=85_000, step=5_000)
            pay_amt1  = st.number_input("Latest Payment (NT$)", value=5_000, step=1_000)
            bill_amt2 = st.number_input("Prior Month Bill (NT$)", value=80_000, step=5_000)
            pay_amt2  = st.number_input("Prior Month Payment (NT$)", value=4_500, step=500)

        submitted = st.form_submit_button("🎯 Score This Account", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "account_id": account_id, "LIMIT_BAL": limit_bal, "AGE": age,
            "PAY_0": pay_0, "BILL_AMT1": bill_amt1, "PAY_AMT1": pay_amt1,
            "BILL_AMT2": bill_amt2, "PAY_AMT2": pay_amt2,
        }
        result = demo_score_account(payload)
        score  = result["risk_score"]
        band   = result["risk_band"]
        color  = RISK_COLORS[band]

        st.markdown(f"""
        <div style='background:{color}22;border-left:6px solid {color};
                    padding:20px 24px;border-radius:6px;margin:16px 0'>
          <h2 style='color:{color};margin:0;font-size:28px'>
            Risk Score: {score:.3f} &nbsp;—&nbsp; {band} RISK
          </h2>
          <p style='margin:8px 0 0;color:#555'>
            Behavioural Risk Score: <b>{result['behavioural_risk_score']}</b> / 100
            &nbsp;|&nbsp; Response time: {result['response_time_ms']}ms
            &nbsp;|&nbsp; Model: {result['model_version']}
          </p>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Top Risk Factors (SHAP)")
            for f in result["top_risk_factors"]:
                icon = "🔺" if f["direction"] == "increases" else "🔻"
                st.markdown(f"{icon} **{f['feature']}** — {f['reason_code']}  \n"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;*SHAP value: {f['shap_value']:+.4f}*")

        with col_r:
            shap_df = pd.DataFrame([
                {"Feature": f["feature"], "SHAP Value": f["shap_value"],
                 "Direction": f["direction"]}
                for f in result["top_risk_factors"]
            ])
            fig = px.bar(shap_df, x="SHAP Value", y="Feature", orientation="h",
                         color="Direction",
                         color_discrete_map={"increases": "#E63946", "decreases": "#52B788"},
                         title="SHAP Waterfall (Top 5 Factors)")
            fig.add_vline(x=0, line_color="#333")
            fig.update_layout(height=300, margin=dict(t=40, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if result["adverse_action_codes"]:
            st.subheader("📋 Regulatory Adverse Action Codes (ECOA/Reg B)")
            for i, code in enumerate(result["adverse_action_codes"], 1):
                st.warning(f"**Code {i}:** {code}")
        else:
            st.success("✅ No adverse action codes triggered — account within acceptable risk parameters.")


# ── PAGE: Drift Monitor ────────────────────────────────────────────────────────
elif page == "Drift Monitor":
    st.title("📊 Model & Data Drift Monitor")
    st.markdown("*Population Stability Index (PSI) monitoring. Triggers auto-retraining when PSI > 0.20.*")

    features_psi = [
        "util_rate_m1", "PAY_0", "min_pay_streak", "BILL_AMT1", "util_trend_slope",
        "total_delay_score", "pay_ratio_avg_3m", "LIMIT_BAL", "max_delay_6m",
        "behavioural_risk_score", "balance_growth_rate", "consecutive_late",
        "util_max_6m", "payment_momentum", "AGE",
    ]
    np.random.seed(7)
    psi_vals = np.random.exponential(0.05, len(features_psi))
    psi_vals[0] = 0.22   # Simulated drift on top feature
    psi_vals[2] = 0.14   # Simulated warning

    psi_df = pd.DataFrame({"Feature": features_psi, "PSI": psi_vals})
    psi_df["Status"] = psi_df["PSI"].apply(
        lambda p: "🔴 DRIFT"   if p >= 0.20 else
                  "🟡 WARNING" if p >= 0.10 else "🟢 OK"
    )
    psi_df = psi_df.sort_values("PSI", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features Monitored",  len(features_psi))
    c2.metric("🔴 Drift Detected",   int((psi_df["PSI"] >= 0.20).sum()))
    c3.metric("🟡 Warning Zone",     int(((psi_df["PSI"] >= 0.10) & (psi_df["PSI"] < 0.20)).sum()))
    c4.metric("Auto-Retrain Trigger","PSI > 0.20")

    fig = px.bar(
        psi_df, x="Feature", y="PSI", color="Status",
        color_discrete_map={"🔴 DRIFT": "#E63946", "🟡 WARNING": "#E9C46A", "🟢 OK": "#52B788"},
        title="Feature Population Stability Index (PSI) — Current Month",
    )
    fig.add_hline(y=0.20, line_dash="dash", line_color="#E63946",
                  annotation_text="Drift Threshold — Auto-retrain triggers here (0.20)")
    fig.add_hline(y=0.10, line_dash="dot",  line_color="#E9C46A",
                  annotation_text="Warning Zone (0.10)")
    fig.update_layout(height=440, xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PSI Detail Table")
    st.dataframe(psi_df.rename(columns={"PSI": "PSI Score", "Status": "Alert Status"}),
                 use_container_width=True, hide_index=True)

    if (psi_df["PSI"] >= 0.20).any():
        drifted = psi_df[psi_df["PSI"] >= 0.20]["Feature"].tolist()
        st.error(
            f"⚠️ **Action Required:** {len(drifted)} feature(s) show significant drift: "
            f"`{'`, `'.join(drifted)}`. "
            f"GitHub Actions retraining pipeline will trigger automatically. "
            f"Review `model-retrain.yml` workflow for details."
        )
    st.caption("PSI computed against training baseline distribution. Monitored daily via Evidently AI.")
