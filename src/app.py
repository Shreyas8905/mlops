import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Risk AI", page_icon="🏦")


@st.cache_resource
def load_assets():
    model = joblib.load('model.joblib')
    encoders = joblib.load('encoders.joblib')
    return model, encoders

try:
    model, encoders = load_assets()
except FileNotFoundError:
    st.error("Error: Model files not found. Please run 'python model.py' first to train the model.")
    st.stop()

st.title("🏦 Loan Risk Predictor")
st.write("Instant credit risk analysis using a pre-trained Random Forest model.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income ($)", 0, 500000, 50000)
        emp_type = st.selectbox("Employment Type", encoders['EmploymentType'].classes_)
        res_type = st.selectbox("Residence Type", encoders['ResidenceType'].classes_)

    with col2:
        credit = st.slider("Credit Score", 300, 850, 650)
        loan_amt = st.number_input("Loan Amount ($)", 0, 500000, 20000)
        term = st.selectbox("Term (Months)", [12, 24, 36, 48, 60])
        default = st.radio("Previous Default?", encoders['PreviousDefault'].classes_)

    submitted = st.form_submit_button("Run Inference", use_container_width=True)

if submitted:
    raw_data = {
        'Age': age, 'Income': income, 'EmploymentType': emp_type,
        'ResidenceType': res_type, 'CreditScore': credit,
        'LoanAmount': loan_amt, 'LoanTerm': term, 'PreviousDefault': default
    }
    input_df = pd.DataFrame([raw_data])

    for col in ['EmploymentType', 'ResidenceType', 'PreviousDefault']:
        input_df[col] = encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    
    st.divider()
    if prediction == "Low Risk":
        st.success(f"Prediction: **{prediction}**")
    elif prediction == "Medium Risk":
        st.warning(f"Prediction: **{prediction}**")
    else:
        st.error(f"Prediction: **{prediction}**")