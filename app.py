
import os
import json
import joblib
import pandas as pd
import streamlit as st
if "page" not in st.session_state:
    st.session_state.page = "Business Problem"

page = st.sidebar.radio(
    "Navigation",
    ["Business Problem", "Data Insights", "Prediction"]
)
if page == "Business Problem":
    st.title("Business Problem")

    st.header("Problem Statement")
    st.write("""
    Banks and lending institutions need to assess the credit risk of loan
    applicants before approving loans. Incorrectly approving a high-risk
    applicant can result in financial losses, while rejecting a good applicant
    can lead to missed business opportunities.
    """)

    st.header("Analytics Objective")
    st.write("""
    The objective is to use applicant and loan-related information to predict
    whether an applicant is likely to be classified as a good or bad credit risk.
    """)

    st.header("Business Decision")
    st.write("""
    The prediction can support lenders in deciding whether an application
    requires standard underwriting or additional credit-risk review.
    """)

elif page == "Data Insights":
    exec(open("pages/pages2_Data_Insights.py.txt").read())
    st.stop()

elif page == "Prediction":
    st.title("Loan Default Risk Prediction")

if page != "Prediction":
    st.stop()

MODEL_PATH = "models/loan_default_model.pkl"
METRICS_PATH = "models/model_metrics.json"

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

st.caption("Business Analytics ML Project | UCI Statlog German Credit Data")

st.info(
    "This application estimates the probability that a loan applicant "
    "will be classified as a bad credit risk. It is an academic decision-support "
    "prototype and should not be used as the sole basis for real lending decisions."
)

if not os.path.exists(MODEL_PATH):
    st.warning(
        "The trained model is not present. Run `python train_model.py` first "
        "to download the UCI dataset, train the models and create the model file."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.header("Applicant Information")
st.sidebar.write("Enter the applicant's information below.")

# UCI German Credit categorical codes are used directly so the application
# remains consistent with the original dataset.
feature_names = [f"feature_{i}" for i in range(1, 21)]

# Friendly controls corresponding to the 20 UCI attributes.
f1 = st.sidebar.selectbox("1. Checking account status", ["A11", "A12", "A13", "A14"])
f2 = st.sidebar.number_input("2. Loan duration (months)", 1, 72, 24)
f3 = st.sidebar.selectbox("3. Credit history", ["A30", "A31", "A32", "A33", "A34"])
f4 = st.sidebar.selectbox("4. Purpose", ["A40", "A41", "A410", "A42", "A43", "A44", "A45", "A46", "A48", "A49"])
f5 = st.sidebar.number_input("5. Credit amount", 250, 20000, 4000)
f6 = st.sidebar.selectbox("6. Savings account/bonds", ["A61", "A62", "A63", "A64", "A65"])
f7 = st.sidebar.selectbox("7. Employment duration", ["A71", "A72", "A73", "A74", "A75"])
f8 = st.sidebar.slider("8. Installment rate (% disposable income)", 1, 4, 2)
f9 = st.sidebar.selectbox("9. Personal status/sex", ["A91", "A92", "A93", "A94"])
f10 = st.sidebar.selectbox("10. Other debtors/guarantors", ["A101", "A102", "A103"])
f11 = st.sidebar.slider("11. Residence since (years)", 1, 4, 2)
f12 = st.sidebar.selectbox("12. Property", ["A121", "A122", "A123", "A124"])
f13 = st.sidebar.slider("13. Age", 18, 75, 35)
f14 = st.sidebar.selectbox("14. Other installment plans", ["A141", "A142", "A143"])
f15 = st.sidebar.selectbox("15. Housing", ["A151", "A152", "A153"])
f16 = st.sidebar.slider("16. Number of existing credits", 1, 4, 1)
f17 = st.sidebar.selectbox("17. Job", ["A171", "A172", "A173", "A174"])
f18 = st.sidebar.slider("18. Number of dependents", 1, 2, 1)
f19 = st.sidebar.selectbox("19. Telephone", ["A191", "A192"])
f20 = st.sidebar.selectbox("20. Foreign worker", ["A201", "A202"])

input_df = pd.DataFrame([[
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
    f11, f12, f13, f14, f15, f16, f17, f18, f19, f20
]], columns=feature_names)

if st.button("Predict Default Risk", type="primary"):
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(model.predict(input_df)[0])

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Default Probability", f"{probability:.1%}")
    c2.metric("Risk Classification", "HIGH RISK" if prediction == 1 else "LOWER RISK")
    c3.metric("Model Output", "Bad Credit" if prediction == 1 else "Good Credit")

    if probability >= 0.60:
        st.error("⚠ High Risk: the model estimates a relatively high probability of default.")
        st.markdown("""
        **Suggested business actions**
        - Conduct enhanced credit assessment.
        - Verify income, existing obligations and repayment capacity.
        - Consider a lower sanctioned amount or stronger security where appropriate.
        - Review whether the loan tenure is consistent with repayment capacity.
        """)
    elif probability >= 0.40:
        st.warning("⚠ Medium Risk: the application should receive additional review.")
        st.markdown("""
        **Suggested business actions**
        - Perform additional affordability and document checks.
        - Compare the application with internal risk policies.
        - Consider risk-adjusted pricing or a conservative credit limit.
        """)
    else:
        st.success("✓ Lower Risk: the model estimates a comparatively lower probability of default.")
        st.markdown("""
        **Suggested business actions**
        - Continue standard underwriting checks.
        - Consider normal approval workflow subject to policy.
        - Monitor repayment behavior after disbursement.
        """)

    st.progress(min(probability, 1.0))
    st.caption("Probability is a model estimate, not a guarantee of future repayment behavior.")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.write("Selected model:", metrics.get("best_model", "N/A"))
    st.sidebar.write("ROC-AUC:", metrics["results"][metrics["best_model"]]["roc_auc"])
