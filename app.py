import streamlit as st
import pandas as pd
import joblib

# تحميل الملفات
model = joblib.load("chickenpox_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Chickenpox Prediction App")

# Inputs
age = st.number_input("Age", 1, 100, 20)
gender = st.selectbox("Gender", ["M", "F"])

fever = st.checkbox("Fever")
rash = st.checkbox("Rash")
itching = st.checkbox("Itching")
fatigue = st.checkbox("Fatigue")
loss = st.checkbox("Loss of Appetite")
headache = st.checkbox("Headache")
muscle = st.checkbox("Muscle Ache")
malaise = st.checkbox("Malaise")

if st.button("Predict"):

    # DataFrame
    sample = pd.DataFrame([{
        "age": age,
        "gender": 1 if gender == "M" else 0,
        "Fever": int(fever),
        "Rash": int(rash),
        "Itching": int(itching),
        "Fatigue": int(fatigue),
        "Loss_of_Appetite": int(loss),
        "Headache": int(headache),
        "Muscle_Ache": int(muscle),
        "Malaise": int(malaise)
    }])

    # Scaling
    sample_scaled = scaler.transform(sample)

    # Prediction
    prediction = model.predict(sample_scaled)[0]

    # Output
    if prediction == 1:
        st.error("Possible Chickenpox ⚠️")
    else:
        st.success("Normal ✅")
