import pandas as pd
import joblib

try:
    print("Loading model and scaler...")
    model = joblib.load("motor_failure_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# Load the new dataset
file_path = "cleaned_data.csv"

try:
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully. First few rows:")
    print(df.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Check if required columns exist
required_columns = {"vibration_sensor", "temperature_sensor"}
if not required_columns.issubset(df.columns):
    print(f"Missing required columns: {required_columns - set(df.columns)}")
    exit()

# Select only feature columns
X_new = df[["vibration_sensor", "temperature_sensor"]]

try:
    print("Scaling features...")
    X_new_scaled = scaler.transform(X_new)
    print("Features scaled successfully.")
except Exception as e:
    print(f"Error during scaling: {e}")
    exit()

# Make predictions
try:
    print("Making predictions...")
    predictions = model.predict(X_new_scaled)
    print("Predictions made successfully.")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# Add predictions to the DataFrame (Fix label mapping)
df["predicted_label"] = predictions
df["predicted_condition"] = df["predicted_label"].map({0: "Normal", 1: "Failure"})  # <-- FIXED HERE

# Save predictions to a new CSV file
output_file = "predictions1.csv"
try:
    df.to_csv(output_file, index=False)
    print(f"Predictions saved in '{output_file}'. Here are the first few predictions:")
    print(df[["vibration_sensor", "temperature_sensor", "predicted_condition"]].head())
except Exception as e:
    print(f"Error saving predictions: {e}")
