import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import CATEGORICAL_OPTIONS, predict_applicant

st.set_page_config(page_title="Predict | EMIPredict AI", page_icon="🔮", layout="wide")
st.title("🔮 Real-Time EMI Prediction")
st.write("Enter an applicant's details to get an instant eligibility decision and "
         "maximum affordable monthly EMI estimate.")

with st.form("applicant_form"):
    st.markdown("#### 👤 Personal Details")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", min_value=21, max_value=65, value=35)
    gender = c2.selectbox("Gender", CATEGORICAL_OPTIONS["gender"])
    marital_status = c3.selectbox("Marital Status", CATEGORICAL_OPTIONS["marital_status"])
    education = c4.selectbox("Education", CATEGORICAL_OPTIONS["education"])

    st.markdown("#### 💼 Employment")
    c1, c2, c3 = st.columns(3)
    employment_type = c1.selectbox("Employment Type", CATEGORICAL_OPTIONS["employment_type"])
    company_type = c2.selectbox("Company Type", CATEGORICAL_OPTIONS["company_type"])
    years_of_employment = c3.number_input("Years of Employment", min_value=0.0, max_value=40.0, value=5.0, step=0.5)

    st.markdown("#### 💰 Income & Housing")
    c1, c2, c3 = st.columns(3)
    monthly_salary = c1.number_input("Monthly Salary (₹)", min_value=5000, max_value=1000000, value=50000, step=1000)
    house_type = c2.selectbox("House Type", CATEGORICAL_OPTIONS["house_type"])
    monthly_rent = c3.number_input("Monthly Rent (₹)", min_value=0, max_value=200000, value=10000, step=500)

    st.markdown("#### 👨‍👩‍👧 Family & Expenses")
    c1, c2, c3, c4 = st.columns(4)
    family_size = c1.number_input("Family Size", min_value=1, max_value=10, value=3)
    dependents = c2.number_input("Dependents", min_value=0, max_value=8, value=1)
    school_fees = c3.number_input("School Fees (₹/mo)", min_value=0, max_value=50000, value=0, step=500)
    college_fees = c4.number_input("College Fees (₹/mo)", min_value=0, max_value=80000, value=0, step=500)

    c1, c2 = st.columns(2)
    travel_expenses = c1.number_input("Travel Expenses (₹/mo)", min_value=0, max_value=30000, value=3000, step=500)
    groceries_utilities = c2.number_input("Groceries & Utilities (₹/mo)", min_value=0, max_value=50000, value=8000, step=500)
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0, max_value=50000, value=2000, step=500)

    st.markdown("#### 🏦 Financial Profile")
    c1, c2, c3, c4 = st.columns(4)
    existing_loans = c1.selectbox("Existing Loans?", CATEGORICAL_OPTIONS["existing_loans"])
    current_emi_amount = c2.number_input("Current EMI Amount (₹)", min_value=0, max_value=100000, value=0, step=500)
    credit_score = c3.number_input("Credit Score", min_value=300, max_value=850, value=700)
    bank_balance = c4.number_input("Bank Balance (₹)", min_value=0, max_value=5000000, value=100000, step=5000)

    emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, max_value=2000000, value=50000, step=5000)

    st.markdown("#### 📝 Requested Loan")
    c1, c2, c3 = st.columns(3)
    emi_scenario = c1.selectbox("EMI Scenario", CATEGORICAL_OPTIONS["emi_scenario"])
    requested_amount = c2.number_input("Requested Amount (₹)", min_value=1000, max_value=5000000, value=200000, step=1000)
    requested_tenure = c3.number_input("Requested Tenure (months)", min_value=1, max_value=84, value=24)

    submitted = st.form_submit_button("🔍 Predict Eligibility", use_container_width=True)

if submitted:
    raw = dict(
        age=age, gender=gender, marital_status=marital_status, education=education,
        monthly_salary=monthly_salary, employment_type=employment_type,
        years_of_employment=years_of_employment, company_type=company_type,
        house_type=house_type, monthly_rent=monthly_rent, family_size=family_size,
        dependents=dependents, school_fees=school_fees, college_fees=college_fees,
        travel_expenses=travel_expenses, groceries_utilities=groceries_utilities,
        other_monthly_expenses=other_monthly_expenses, existing_loans=existing_loans,
        current_emi_amount=current_emi_amount, credit_score=credit_score,
        bank_balance=bank_balance, emergency_fund=emergency_fund,
        emi_scenario=emi_scenario, requested_amount=requested_amount,
        requested_tenure=requested_tenure,
    )

    result = predict_applicant(raw)

    st.markdown("---")
    st.markdown("### 📋 Prediction Result")

    color_map = {"Eligible": "🟢", "High_Risk": "🟡", "Not_Eligible": "🔴"}
    r1, r2 = st.columns([1, 1.3])

    with r1:
        st.markdown(f"## {color_map.get(result['eligibility'], '')} {result['eligibility'].replace('_', ' ')}")
        st.metric("Estimated Max Monthly EMI", f"₹{result['max_monthly_emi']:,.0f}")

        est_emi = requested_amount / requested_tenure
        if est_emi <= result["max_monthly_emi"]:
            st.success(f"Requested EMI (~₹{est_emi:,.0f}/mo) is within the estimated affordability limit.")
        else:
            st.warning(f"Requested EMI (~₹{est_emi:,.0f}/mo) exceeds the estimated affordability limit "
                       f"of ₹{result['max_monthly_emi']:,.0f}/mo.")

    with r2:
        probs = result["probabilities"]
        fig = go.Figure(go.Bar(
            x=list(probs.values()), y=list(probs.keys()), orientation="h",
            marker_color=["#2ecc71" if k == "Eligible" else "#f39c12" if k == "High_Risk" else "#e74c3c"
                          for k in probs.keys()],
            text=[f"{v*100:.1f}%" for v in probs.values()], textposition="outside",
        ))
        fig.update_layout(title="Class Probabilities", xaxis_title="Probability",
                           xaxis_range=[0, 1], height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 View engineered features used for this prediction"):
        feat_df = pd.DataFrame(result["features"].items(), columns=["Feature", "Value"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
