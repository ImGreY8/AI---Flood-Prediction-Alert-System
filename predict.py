import joblib
import numpy as np
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "flood_model.pkl")
model = joblib.load(model_path)


def predict_flood(data):
    features = np.array([
        data['rainfall'],
        data['river_level'],
        data['humidity'],
        data['temperature']
    ]).reshape(1, -1)

    prob = model.predict_proba(features)[0][1]
    result = {
        "flood_risk_percent": round(prob * 100, 2),
        "flood_prediction": "High Risk" if prob > 0.5 else "Low Risk"
    }
    return result
