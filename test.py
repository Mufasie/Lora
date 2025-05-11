import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load("robust_motor_status_model3.pkl")

# Sample test data: each row is [voltage, current]
# You can modify or expand this list to test with more data
test_samples = np.array([
    [9.14, 0.93],   # Expected: 1 (Load fault)
    [8.90, 0.84],   # Expected: 2 (Battery fault)
    [7.90, 0.90],   # Expected: 3 (Both faults)
    [9.1, 0.13],    # Expected: 0 (Normal)
])

# Convert the test samples to a DataFrame with appropriate column names
test_samples_df = pd.DataFrame(test_samples, columns=['voltage', 'current'])

# Predict using the loaded model
predictions = model.predict(test_samples_df)

# Print results
for i, (sample, pred) in enumerate(zip(test_samples_df.values, predictions), start=1):
    print(f"Sample {i}: Voltage={sample[0]}, Current={sample[1]} → Predicted Label: {pred}")
