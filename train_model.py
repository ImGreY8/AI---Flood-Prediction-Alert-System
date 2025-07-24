from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

# Load dataset
df = pd.read_excel("flood_sample_data.xlsx")

# Features and labels
X = df[['rainfall', 'river_level', 'humidity', 'temperature']]
y = df['flood']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
model_path = os.path.join(os.path.dirname(__file__), "flood_model.pkl")
joblib.dump(model, model_path)

print("✅ Model trained and saved as flood_model.pkl")
print("testing done")
