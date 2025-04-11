import streamlit as st
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Load model and binarizer
with open('disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

# Get list of all symptoms
all_symptoms = mlb.classes_

st.title("🩺 Disease Predictor")
st.write("Select the symptoms you're experiencing:")

# Multiselect widget
selected_symptoms = st.multiselect("Choose symptoms", all_symptoms)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Transform the input
        input_data = mlb.transform([selected_symptoms])

        # Predict
        prediction = model.predict(input_data)

        st.success(f"🧬 Predicted Disease: **{prediction[0]}**")
