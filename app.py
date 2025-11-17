import streamlit as st
import pandas as pd
import joblib

st.title("Student Depression Classification")

model = joblib.load("model.pkl")

map_level = {"Low": 1, "Medium": 2, "High": 3}

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=40)
academic_pressure = st.selectbox("Academic Pressure", ["Low", "Medium", "High"])
work_pressure = st.selectbox("Work Pressure", ["Low", "Medium", "High"])
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
study_satisfaction = st.selectbox("Study Satisfaction", ["Low", "Medium", "High"])
job_satisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High"])

sleep_duration = st.selectbox("Sleep Duration", [
    "Less than 5 hours", "5-6 hours", "7-8 hours",
    "More than 8 hours", "Others"
])
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Others"])
degree = st.selectbox("Degree", [
    "Class 12", "B.Arch", "B.Com", "B.Ed", "B.Pharm",
    "B.Tech", "BA", "BBA", "BCA", "BE", "BHM", "BSc",
    "LLB", "LLM", "M.Com", "M.Ed", "M.Pharm", "M.Tech",
    "MA", "MBA", "MBBS", "MCA", "MD", "ME", "MHM",
    "MSc", "Others", "PhD"
])
suicidal = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
work_study_hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=18.0)
financial_stress = st.selectbox("Financial Stress", ["1.0", "2.0", "3.0", "4.0", "5.0", "?"])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

df_input = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Academic Pressure": [map_level[academic_pressure]],
    "Work Pressure": [map_level[work_pressure]],
    "CGPA": [cgpa],
    "Study Satisfaction": [map_level[study_satisfaction]],
    "Job Satisfaction": [map_level[job_satisfaction]],
    "Sleep Duration": [sleep_duration],
    "Dietary Habits": [dietary_habits],
    "Degree": [degree],
    "Have you ever had suicidal thoughts ?": [suicidal],
    "Work/Study Hours": [work_study_hours],
    "Financial Stress": [financial_stress],
    "Family History of Mental Illness": [family_history]
})

if st.button("Prediksi"):
    pred = model.predict(df_input)[0]

    if pred == 0:
        st.success("Hasil: **Tidak Depresi**")
    else:
        st.error("Hasil: **Depresi**")
