import streamlit as st
import pandas as pd
import pickle

st.title("Student Depression Classification")

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    st.error("Model 'model.pkl' tidak ditemukan atau tidak bisa dibaca.")
    st.stop()

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=10, max_value=40, step=1)
academic_pressure = st.selectbox("Academic Pressure", ["Low", "Medium", "High"])
work_pressure = st.selectbox("Work Pressure", ["Low", "Medium", "High"])
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
study_satisfaction = st.selectbox("Study Satisfaction", ["Low", "Medium", "High"])
job_satisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High"])
sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours"])
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
degree = st.selectbox("Degree", ["Bachelor", "Masters", "PhD"])
suicidal = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
work_study_hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=18.0, step=0.5)
financial_stress = st.selectbox("Financial Stress", ["Low", "Medium", "High"])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

df_input = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Academic Pressure": [academic_pressure],
    "Work Pressure": [work_pressure],
    "CGPA": [cgpa],
    "Study Satisfaction": [study_satisfaction],
    "Job Satisfaction": [job_satisfaction],
    "Sleep Duration": [sleep_duration],
    "Dietary Habits": [dietary_habits],
    "Degree": [degree],
    "Have you ever had suicidal thoughts ?": [suicidal],
    "Work/Study Hours": [work_study_hours],
    "Financial Stress": [financial_stress],
    "Family History of Mental Illness": [family_history]
})

if st.button("Prediksi"):
    try:
        pred = model.predict(df_input)[0]
        st.subheader("Hasil Prediksi:")
        if pred == 1:
            st.error("Mahasiswa terindikasi mengalami Depression")
        else:
            st.success("Mahasiswa tidak terindikasi Depression")
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
