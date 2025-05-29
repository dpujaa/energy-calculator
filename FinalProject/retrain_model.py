import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# === STEP 1: Load dataset ===
df = pd.read_csv("final_dataset_by_zip.csv")
df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

# === STEP 2: Create a better target ===
# Monthly kWh per household = Total Usage (annual) / 12 months / avg household size
df["monthly_kwh_per_household"] = (df["Total Usage"] / 12) / df["avg_household_size"]

# === STEP 3: Melt temperature data to long format ===
weather_cols = [f"temp_month_{i}" for i in range(1, 13)]
long_df = df.melt(
    id_vars=["zip_code", "monthly_kwh_per_household"],
    value_vars=weather_cols,
    var_name="month",
    value_name="temperature"
)

# Extract month number from "temp_month_X"
long_df["month"] = long_df["month"].str.extract("(\d+)").astype(int)

# === STEP 4: Merge back to get full training data ===
# Merge in household size so we can scale back up later
long_df = long_df.merge(df[["zip_code", "avg_household_size"]], on="zip_code")

# Features and target
X = long_df[["temperature"]]
y = long_df["monthly_kwh_per_household"]

# === STEP 5: Train/Test split and train model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# === STEP 6: Save model ===
joblib.dump(model, "electricity_bill_predictor.pkl")
print("✅ Smarter model saved as 'electricity_bill_predictor.pkl'")
