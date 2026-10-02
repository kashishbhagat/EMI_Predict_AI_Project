import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_features_sample

st.set_page_config(page_title="Data Exploration | EMIPredict AI", page_icon="📊", layout="wide")
st.title("📊 Data Exploration")

df = load_features_sample(60000)
st.caption(f"Showing an analysis based on a random sample of {len(df):,} of the 404,792 cleaned records "
           "(full dataset used for model training).")

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Eligibility Rate", f"{(df['emi_eligibility']=='Eligible').mean()*100:.1f}%")
k2.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")
k3.metric("Avg Monthly Salary", f"₹{df['monthly_salary'].mean():,.0f}")
k4.metric("Avg Max Monthly EMI", f"₹{df['max_monthly_emi'].mean():,.0f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Eligibility Overview", "Financial Ratios", "Demographics", "Correlations"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        counts = df["emi_eligibility"].value_counts().reset_index()
        counts.columns = ["Eligibility", "Count"]
        fig = px.bar(counts, x="Eligibility", y="Count", color="Eligibility",
                     color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"},
                     title="EMI Eligibility Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ct = pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index") * 100
        ct = ct.reset_index().melt(id_vars="emi_scenario", var_name="Eligibility", value_name="Percent")
        fig = px.bar(ct, x="emi_scenario", y="Percent", color="Eligibility", barmode="stack",
                     color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"},
                     title="Eligibility Rate by EMI Scenario")
        fig.update_xaxes(tickangle=20)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="max_monthly_emi", nbins=60, color_discrete_sequence=["teal"],
                        title="Distribution of Maximum Monthly EMI (Regression Target)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="emi_eligibility", y="obligation_to_income_ratio", color="emi_eligibility",
                     color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"},
                     title="Obligation-to-Income Ratio by Eligibility")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, x="emi_eligibility", y="credit_score", color="emi_eligibility",
                     color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"},
                     title="Credit Score by Eligibility")
        st.plotly_chart(fig, use_container_width=True)

    sample = df.sample(min(15000, len(df)), random_state=1)
    fig = px.scatter(sample, x="requested_amount", y="max_monthly_emi", color="emi_scenario",
                      opacity=0.4, title="Requested Amount vs Max Monthly EMI")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="age", nbins=30, color_discrete_sequence=["slateblue"],
                            title="Applicant Age Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ct = pd.crosstab(df["employment_type"], df["emi_eligibility"], normalize="index") * 100
        ct = ct.reset_index().melt(id_vars="employment_type", var_name="Eligibility", value_name="Percent")
        fig = px.bar(ct, x="employment_type", y="Percent", color="Eligibility", barmode="stack",
                     color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"},
                     title="Eligibility Rate by Employment Type")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    key_cols = [
        "monthly_salary", "credit_score", "debt_to_income_ratio",
        "obligation_to_income_ratio", "disposable_income_ratio",
        "affordability_risk_score", "emergency_fund_to_salary_ratio",
        "requested_amount", "max_monthly_emi",
    ]
    corr = df[key_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation: Key Financial Variables")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Key insights:**")
    st.markdown(f"""
- Overall eligibility rate across the dataset is **18.4%**.
- E-commerce Shopping EMI and Home Appliances EMI have the highest approval rates (~26%),
  while Vehicle and Personal Loan EMI have the lowest (~11%).
- `obligation_to_income_ratio` is negatively correlated with `max_monthly_emi`
  (r ≈ -0.43) — the more of their income an applicant already commits to
  expenses/EMIs, the lower their affordable EMI headroom.
- `credit_score` is positively correlated with `max_monthly_emi` (r ≈ 0.28).
""")
