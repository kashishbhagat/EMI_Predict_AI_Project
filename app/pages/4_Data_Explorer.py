import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import load_cleaned_sample

st.set_page_config(page_title="Data Explorer | EMIPredict AI", page_icon="🗂️", layout="wide")
st.title("🗂️ Applicant Data Explorer")
st.write("Browse, filter, and export the cleaned applicant dataset.")

df = load_cleaned_sample(60000)

with st.sidebar:
    st.markdown("### 🔍 Filters")
    scenario_filter = st.multiselect("EMI Scenario", sorted(df["emi_scenario"].unique()))
    eligibility_filter = st.multiselect("Eligibility", sorted(df["emi_eligibility"].unique()))
    age_range = st.slider("Age", int(df["age"].min()), int(df["age"].max()),
                           (int(df["age"].min()), int(df["age"].max())))
    credit_range = st.slider("Credit Score", 300, 850, (300, 850))
    salary_range = st.slider("Monthly Salary (₹)", int(df["monthly_salary"].min()),
                              int(df["monthly_salary"].max()),
                              (int(df["monthly_salary"].min()), int(df["monthly_salary"].max())))

filtered = df.copy()
if scenario_filter:
    filtered = filtered[filtered["emi_scenario"].isin(scenario_filter)]
if eligibility_filter:
    filtered = filtered[filtered["emi_eligibility"].isin(eligibility_filter)]
filtered = filtered[
    filtered["age"].between(*age_range)
    & filtered["credit_score"].between(*credit_range)
    & filtered["monthly_salary"].between(*salary_range)
]

st.markdown(f"**{len(filtered):,} records match your filters** (of {len(df):,} sampled)")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Credit Score", f"{filtered['credit_score'].mean():.0f}" if len(filtered) else "—")
col2.metric("Avg Salary", f"₹{filtered['monthly_salary'].mean():,.0f}" if len(filtered) else "—")
col3.metric("Eligible %", f"{(filtered['emi_eligibility']=='Eligible').mean()*100:.1f}%" if len(filtered) else "—")

st.dataframe(filtered, use_container_width=True, height=500)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered data as CSV", csv, "emi_filtered_data.csv", "text/csv")
