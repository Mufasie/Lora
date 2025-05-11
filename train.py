import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
import joblib

# Load data
df = pd.read_csv("newData2.csv")

# Features and target (only using 'voltage' and 'current')
features = ['voltage', 'current']
X = df[features]
y = df['label']

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing: Scale robustly to handle outliers
preprocessor = ColumnTransformer([
    ('scaler', RobustScaler(), features)
])

# Pipeline with preprocessing + classifier
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: cross-validation score
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print("Cross-validation Accuracy Scores:", cv_scores)
print("Average CV Accuracy:", np.mean(cv_scores))

# Save the pipeline model
joblib.dump(pipeline, "robust_motor_status_model3.pkl")
