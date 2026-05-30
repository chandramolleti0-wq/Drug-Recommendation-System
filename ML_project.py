import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Sample Dataset (Replace with your CSV later)
# -----------------------------
data = {
    "Fever": [1,1,0,0,1,0],
    "Cough": [1,0,1,0,1,0],
    "Headache": [1,0,0,1,1,0],
    "Fatigue": [1,1,0,1,1,0],
    "Disease": ["Flu","Viral Fever","Cold","Migraine","Flu","Healthy"],
    "Drug": ["Paracetamol","Dolo 650","Cetirizine","Sumatriptan","Paracetamol","None"],
    "Description": [
        "Flu is a viral infection.",
        "Viral fever caused by infection.",
        "Common cold infection.",
        "Migraine is severe headache.",
        "Flu is a viral infection.",
        "No disease detected."
    ],
    "Precaution": [
        "Rest and drink fluids",
        "Take proper rest",
        "Avoid cold food",
        "Avoid stress",
        "Rest and drink fluids",
        "Stay healthy"
    ],
    "Diet": [
        "Light food",
        "Soft diet",
        "Warm soup",
        "Healthy diet",
        "Light food",
        "Balanced diet"
    ],
    "Workout": [
        "No workout",
        "No workout",
        "Light yoga",
        "Meditation",
        "No workout",
        "Regular exercise"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Model Training
# -----------------------------
X = df[["Fever","Cough","Headache","Fatigue"]]
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

model = DecisionTreeClassifier()
model.fit(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💊 Drug Recommendation System")

st.write("Select your symptoms:")

fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
headache = st.checkbox("Headache")
fatigue = st.checkbox("Fatigue")

if st.button("Predict"):

    input_data = np.array([[fever, cough, headache, fatigue]])
    prediction = model.predict(input_data)
    disease = le.inverse_transform(prediction)[0]

    result = df[df["Disease"] == disease].iloc[0]

    st.success(f"Predicted Disease: {disease}")
    st.info(f"Recommended Drug: {result['Drug']}")
    st.write("### Description")
    st.write(result["Description"])
    st.write("### Precaution")
    st.write(result["Precaution"])
    st.write("### Diet")
    st.write(result["Diet"])
    st.write("### Workout")
    st.write(result["Workout"])
    