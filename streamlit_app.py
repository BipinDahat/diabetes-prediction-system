import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_xgb.pkl")
scaler = joblib.load("scaler.pkl")

# Page setup
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# Title
st.markdown(
    """
    <h1 style="text-align:center; color:#4CAF50;">
        🩺 Diabetes Prediction System
    </h1>
    <p style="text-align:center; font-size:18px;">
        Use the sliders to enter patient information and predict diabetes.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Two-column layout
col1, col2 = st.columns(2)

# LEFT COLUMN SLIDERS
with col1:
    pregnancies = st.slider(
        "Pregnancies", 0, 20, 1,
        help="Number of pregnancies the patient had."
    )

    glucose = st.slider(
        "Glucose Level", 0, 300, 120,
        help="Plasma glucose concentration."
    )

    blood_pressure = st.slider(
        "Blood Pressure", 0, 200, 70,
        help="Diastolic blood pressure (mm Hg)."
    )

    skin_thickness = st.slider(
        "Skin Thickness", 0, 100, 20,
        help="Triceps skinfold thickness (mm)."
    )

# RIGHT COLUMN SLIDERS
with col2:
    insulin = st.slider(
        "Insulin Level", 0, 900, 80,
        help="2-Hour serum insulin level."
    )

    bmi = st.slider(
        "BMI", 0.0, 70.0, 25.0,
        help="Body Mass Index."
    )

    dpf = st.slider(
        "Diabetes Pedigree Function", 0.0, 3.0, 0.5,
        help="Likelihood of diabetes based on family history."
    )

    age = st.slider(
        "Age", 1, 120, 30,
        help="Age of the patient."
    )

# Combine input
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

# Predict Button Centered
st.markdown("<br>", unsafe_allow_html=True)
center_btn = st.columns([1, 2, 1])[1]

with center_btn:
    if st.button("🔍 Predict Diabetes", use_container_width=True):
        user_scaled = scaler.transform(user_input)
        prediction = model.predict(user_scaled)[0]

        if prediction == 1:
            st.markdown(
                """
                <div style="padding:20px; border-radius:10px; background-color:#ffcccc;">
                    <h3 style="color:#b30000;">⚠️ Result: High Risk of Diabetes</h3>
                    <p>The patient is likely diabetic.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="padding:20px; border-radius:10px; background-color:#ccffcc;">
                    <h3 style="color:#006600;">✅ Result: Low Risk of Diabetes</h3>
                    <p>The patient is likely non-diabetic.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
