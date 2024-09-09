from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # Import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the pre-trained model and scaler for kidney disease
model = joblib.load('kidney_save.joblib')
scaler = joblib.load('scaler_kidney.joblib')  # Ensure the scaler is saved and loaded

# Define the predict route
@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():
    try:
        data = request.json
        age = float(data.get('age', 0))
        bp = float(data.get('bp', 0))
        sg = float(data.get('sg', 0))
        al = float(data.get('al', 0))
        su = float(data.get('su', 0))
        rbc = float(data.get('rbc', 0))
        pc = float(data.get('pc', 0))
        pcc = float(data.get('pcc', 0))
        ba = float(data.get('ba', 0))
        bgr = float(data.get('bgr', 0))
        bu = float(data.get('bu', 0))
        sc = float(data.get('sc', 0))
        sod = float(data.get('sod', 0))
        pot = float(data.get('pot', 0))
        hemo = float(data.get('hemo', 0))
        pcv = float(data.get('pcv', 0))
        wc = float(data.get('wc', 0))
        rc = float(data.get('rc', 0))
        htn = float(data.get('htn', 0))
        dm = float(data.get('dm', 0))
        cad = float(data.get('cad', 0))
        appet = float(data.get('appet', 0))
        pe = float(data.get('pe', 0))
        ane = float(data.get('ane', 0))
        
        input_features = [
            age, bp, sg, al, su, rbc, pc, pcc, ba, bgr,
            bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm,
            cad, appet, pe, ane
        ]
        
        # Create DataFrame and standardize
        input_df = pd.DataFrame([input_features], columns=[
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
            'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm',
            'cad', 'appet', 'pe', 'ane'
        ])
        input_df = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_df)
        return jsonify({'prediction': 'Kidney Disease' if prediction[0] == 1 else 'No Kidney Disease'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
