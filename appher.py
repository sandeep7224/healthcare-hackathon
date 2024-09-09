from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Initialize the Flask app
appher = Flask(__name__)
CORS(appher)  # Enable CORS

# Load the pre-trained model and scaler
model = joblib.load('heart_save.joblib')
scaler = joblib.load('scaler_heart.joblib')  # Ensure the scaler is saved and loaded

# Define the predict route
@appher.route('/heart_predict', methods=['POST'])
def predict_heart():
    try:
        data = request.json
        age = float(data.get('age', 0))
        sex = float(data.get('sex', 0))
        cp = float(data.get('cp', 0))
        trestbps = float(data.get('trestbps', 0))
        chol = float(data.get('chol', 0))
        fbs = float(data.get('fbs', 0))
        restecg = float(data.get('restecg', 0))
        thalach = float(data.get('thalach', 0))
        exang = float(data.get('exang', 0))
        oldpeak = float(data.get('oldpeak', 0))
        slope = float(data.get('slope', 0))
        ca = float(data.get('ca', 0))
        thal = float(data.get('thal', 0))
       
        input_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        input_df = pd.DataFrame([input_features], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        input_df = scaler.transform(input_df)
        
        prediction = model.predict(input_df)
        return jsonify({'prediction': 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    appher.run(debug=True)
