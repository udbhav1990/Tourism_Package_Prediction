import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub 
model_path = hf_hub_download(
    repo_id="udbhav90/tourism-wellness-model",
    filename="best_tourism_model_v1.joblib"
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Wellness Package Purchase Prediction
st.title("Wellness Tourism Package — Purchase Prediction")
st.write(
    "Internal tool to predict whether a customer is likely to purchase the **Wellness Tourism Package**."
)
st.write("Enter customer details and interactions, then click **Predict**.")

# -----------------------------
# Collect user input
# -----------------------------
TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    ["salaried", "freelancer", "business", "student", "other"]
)

Gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["single", "married", "divorced", "widowed"]
)

Designation = st.selectbox(
    "Designation",
    ["executive", "manager", "senior manager", "vp", "director", "other"]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["basic", "deluxe", "super deluxe", "standard"]
)

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=1, value=2)
PreferredPropertyStar = st.number_input("Preferred Property Star (e.g., 3/4/5)", min_value=1, max_value=7, value=4)
NumberOfTrips = st.number_input("Average Number of Trips (per year)", min_value=0, value=2)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting (under 5)", min_value=0, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0)

Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=7)
NumberOfFollowups = st.number_input("Number Of Follow-ups", min_value=0, value=2)
DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, value=15)

# Convert to model-ready single-row DataFrame
input_data = pd.DataFrame([{
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched,
    "Age": Age,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
}])

# Classification threshold (kept same as training script)
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    try:
        proba = model.predict_proba(input_data)[0, 1]
        pred = int(proba >= classification_threshold)
        label = "LIKELY TO PURCHASE" if pred == 1 else "NOT LIKELY TO PURCHASE"

        st.subheader("Prediction")
        st.write(f"**Result:** {label}")
        st.write(f"**Purchase Probability:** {proba:.3f}")
        st.caption("Threshold used: 0.45")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
