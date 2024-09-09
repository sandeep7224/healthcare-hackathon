from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # Import CORS

# Initialize the Flask app
app1 = Flask(__name__)
CORS(app1)  # Enable CORS

# Load the pre-trained model and scaler
model = joblib.load('diabetes.joblib')
scaler = joblib.load('scaler.joblib')  # Ensure the scaler is saved and loaded

# Define the predict route
@app1.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        pregnancies = float(data.get('pregnancies', 0))
        glucose = float(data.get('glucose', 0))
        bloodPressure = float(data.get('bloodPressure', 0))
        skinThickness = float(data.get('skinThickness', 0))
        insulin = float(data.get('insulin', 0))
        bmi = float(data.get('bmi', 0))
        diabetesPedigreeFunction = float(data.get('diabetesPedigreeFunction', 0))
        age = float(data.get('age', 0))
        
        input_features = [pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]
        input_df = pd.DataFrame([input_features], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        input_df = scaler.transform(input_df)
        
        prediction = model.predict(input_df)
        return jsonify({'prediction': 'Diabetes' if prediction[0] == 1 else 'No Diabetes'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app1.run(debug=True)
