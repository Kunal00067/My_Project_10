# Import required libraries
# -----------------------------

import streamlit as st                # For building web app / user interface
import pandas as pd                   # For data manipulation
import numpy as np                    # For numerical computations
import joblib                         # For saving and loading trained ML models
import plotly.graph_objects as go     # For interactive plots

# Custom CSS for styling
# -----------------------------
st.markdown(
    """
    <style>
    main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f7764;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Function to load trained model and scaler
# -----------------------------
@st.cache_resource
def load_model_and_scaler():
    """
    Load the trained model and StandardScaler from .pkl files.
    Returns:
        model: Trained ML model
        scaler: StandardScaler object
    """
    try:
        model = joblib.load('diabetes_model.pkl')   # Load SVM/Random Forest model
        scaler = joblib.load('scaler_svm.pkl')      # Load scaler used during training
        return model, scaler
    except FileNotFoundError:
        # Return None if model/scaler files not found
        return None, None
    
    # -----------------------------
# App Header
# -----------------------------
st.title("Diabetes Prediction System")
st.markdown("### AI-Powered Risk Assessment Tool")

# Load model and scaler
# -----------------------------
model, scaler = load_model_and_scaler()

# Check if model/scaler exist
# -----------------------------
if model is None or scaler is None:
    st.error("⚠️ Model files not found!")
    st.info("""
    Please run the following command first:

    ```bash
    python diabetes_prediction.py
    ```

    This will train and save the model files.
    """)
    st.stop()  # Stop execution if files are missing

    # -----------------------------
# Sidebar Inputs for Patient Information
# -----------------------------

st.sidebar.title("Patient Information")

# Demographics
# -----------------------------
st.sidebar.subheader("Demographics")
age = st.sidebar.slider('Age', min_value=21, max_value=100, value=30)
pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=0)

# -----------------------------
# Medical Measurements
# -----------------------------
st.sidebar.subheader("Medical Measurements")
glucose = st.sidebar.slider('Glucose (mg/dL)', min_value=0, max_value=200, value=120)
bp = st.sidebar.slider('Blood Pressure (mm Hg)', min_value=0, max_value=130, value=70)
skin = st.sidebar.slider('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
insulin = st.sidebar.slider('Insulin (mu U/ml)', min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.sidebar.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, step=0.01)

# -----------------------------
# Predict Button
# -----------------------------
st.sidebar.markdown("---")  # Separator line
predict_btn = st.sidebar.button("Predict", type="primary", use_container_width=True)

# -----------------------------
# Main content: Prepare input for prediction
# -----------------------------
if predict_btn:
    # Convert all sidebar inputs into a NumPy array
    # Shape must be (1, 8) because model expects 8 features in one row
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    st.write("Input Data Prepared for Prediction:")
    st.write(input_data)



    # -----------------------------
# Prediction & Display Results
# -----------------------------

# Standardize input using the trained scaler
input_std = scaler.transform(input_data)  # input_data shape: (1,8)

# Predict diabetes using the trained model
prediction = model.predict(input_std)[0]  # 0 = Non-Diabetic, 1 = Diabetic

# Get probability (if model supports predict_proba)
try:
    probability = model.predict_proba(input_std)[0]  # Returns [prob_negative, prob_positive]
    prob_negative = probability[0] * 100  # Probability for Non-Diabetic
    prob_positive = probability[1] * 100  # Probability for Diabetic
except AttributeError:
    # If model doesn't have predict_proba, assign 100% for predicted class
    prob_positive = 100 if prediction == 1 else 0
    prob_negative = 100 - prob_positive

# -----------------------------
# Display Prediction Results
# -----------------------------
st.markdown("---")  # Horizontal line
st.header("Prediction Results")

# Create two columns for results and gauge
col1, col2 = st.columns([2, 1])

with col1:
    # Display prediction risk
    if prediction == 0:  # Non-Diabetic
        if prob_positive < 30:
            st.success("✅ LOW RISK - Not Diabetic")
        else:
            st.warning("⚠️ MODERATE RISK - Not Diabetic")
    else:  # Diabetic
        if prob_positive > 70:
            st.error("❌ HIGH RISK - Diabetic")
        else:
            st.warning("⚠️ MODERATE RISK - Diabetic")

    # Display probability breakdown
    st.subheader("Probability Breakdown")
    prob_col1, prob_col2 = st.columns(2)
    prob_col1.metric("Non-Diabetic", f"{prob_negative:.1f}%")
    prob_col2.metric("Diabetic", f"{prob_positive:.1f}%")

with col2:
    # Gauge chart for visualizing risk
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_positive,
        title={"text": "Risk Level"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))

    # Adjust layout of gauge
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

    # Display gauge chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Risk Factor Analysis
# -----------------------------

st.subheader("Risk Factor Analysis")

# Initialize lists
risk_factors = []
positive_factors = []

# -----------------------------
# Glucose
# -----------------------------
if glucose > 125:
    risk_factors.append("High Glucose Level (>125 mg/dL)")
elif glucose < 100:
    positive_factors.append("Normal Glucose Level")

# -----------------------------
# BMI
# -----------------------------
if bmi > 30:
    risk_factors.append("High BMI - Obesity (>30)")
elif 18.5 <= bmi <= 24.9:
    positive_factors.append("Healthy BMI (18.5-24.9)")

# -----------------------------
# Age
# -----------------------------
if age > 45:
    risk_factors.append("Age Factor (>45)")

# -----------------------------
# Blood Pressure
# -----------------------------
if bp > 80:
    risk_factors.append("High Blood Pressure (>80 mm Hg)")
elif 60 <= bp <= 80:
    positive_factors.append("Normal Blood Pressure (60-80 mm Hg)")

# -----------------------------
# Diabetes Pedigree Function (Genetic Risk)
# -----------------------------
if dpf > 0.5:
    risk_factors.append("Higher Genetic Predisposition (DPF > 0.5)")

# -----------------------------
# Display Risk Factors
# -----------------------------
if risk_factors:
    st.warning("**Identified Risk Factors:**")
    for factor in risk_factors:
        st.markdown(f"- {factor}")

# -----------------------------
# Display Positive Indicators
# -----------------------------
if positive_factors:
    st.success("**Positive Health Indicators:**")
    for factor in positive_factors:
        st.markdown(f"- {factor}")


# -----------------------------
# Recommendations & Disclaimer
# -----------------------------
st.markdown("---")  # Horizontal line
st.subheader("Recommendations")

if prediction == 1:  # Patient predicted as Diabetic
    st.error("""
**Important Actions:**
- Consult a healthcare professional immediately
- Get comprehensive diabetes screening
- Monitor blood glucose regularly
- Consider lifestyle modifications
""")
else:  # Patient predicted as Non-Diabetic
    st.success("""
**Maintain Healthy Practices:**
- Regular health check-ups
- Balanced diet
- Exercise regularly (30+ min daily)
- Monitor weight and BMI
""")

# -----------------------------
# Medical Disclaimer
# -----------------------------
st.markdown("---")
st.warning("""
**MEDICAL DISCLAIMER**  
This prediction is for educational purposes only. It should NOT replace professional medical advice. Always consult qualified healthcare professionals for medical concerns.
""")

# -----------------------------
# Initial Page / Info when no prediction yet
# -----------------------------
if not predict_btn:
    st.markdown("---")
    st.info("Enter patient information in the sidebar and click *Predict* to see results.")

    # Display model info in three columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "SVM")
    col2.metric("Accuracy", "~78%")  # Update with actual model accuracy
    col3.metric("Dataset", "768 samples")
