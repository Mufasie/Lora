import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import traceback

try:
    print("Loading dataset...")
    file_path = "temp and vib normal and failure.csv"  # Update with actual file path
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")

    print("Reversing labels...")
    df["label"] = df["label"].map({0: 1, 1: 0})
    print("Labels reversed.")

    print("Preparing features and target...")
    X = df[["vibration_sensor", "temperature_sensor"]]
    y = df["label"]
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset split done.")

    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Normalization complete.")

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)

    print("Saving model...")
    joblib.dump(model, "motor_failure_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved successfully.")

except Exception as e:
    print("An error occurred:", e)
    traceback.print_exc()
