from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "flood_model.pkl")
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        rainfall = float(request.form['rainfall'])
        river_level = float(request.form['river_level'])
        humidity = float(request.form['humidity'])
        temperature = float(request.form['temperature'])


        # testing
        # Prepare data for prediction
        features = np.array([[rainfall, river_level, humidity, temperature]])
        prediction = model.predict(features)[0]

        result = "Flood Risk: YES" if prediction == 1 else "Flood Risk: NO"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
