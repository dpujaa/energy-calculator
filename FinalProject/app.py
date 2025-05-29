import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model and dataset ===
model = joblib.load("electricity_bill_predictor2.pkl")
df = pd.read_csv("final_dataset_by_zip.csv")

# Clean and prepare ZIP code
df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

# === Streamlit UI ===
st.title("🔌 Improved Electricity Bill Predictor")

# === User Inputs ===
zip_code = st.text_input("Enter your ZIP Code", value="95616").strip().zfill(5)
household_size = st.number_input("Your Household Size", min_value=1.0, step=1.0, value=3.0)
month = st.slider("Select Month (1 = Jan, 12 = Dec)", min_value=1, max_value=12, value=7)
rate = st.number_input("Your Electricity Rate ($/kWh)", min_value=0.01, step=0.01, value=0.30)

# === Prediction Logic ===
if st.button("Predict My Bill"):
    row = df[df["zip_code"] == zip_code]

    if row.empty:
        st.error("❌ ZIP code not found in dataset.")
    else:
        try:
            temperature = row[f"temp_month_{month}"].values[0]
            input_array = np.array([[temperature, month, household_size]])

            # Predict monthly usage
            predicted_kwh = model.predict(input_array)[0]

            # Optionally clip unrealistic extremes
            predicted_kwh = np.clip(predicted_kwh, 100, 1500)

            # Calculate estimated bill
            bill = predicted_kwh * rate

            st.success(f"📊 Predicted Usage: {predicted_kwh:.0f} kWh")
            st.success(f"💵 Estimated Bill: ${bill:.2f}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
