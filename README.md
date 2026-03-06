# Healthcare Hackathon 🏥

A comprehensive machine learning solution for multiple disease prediction using Python and scikit-learn. This project includes predictive models for diabetes, heart disease, brain tumors, and kidney disease.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Disease Models](#disease-models)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Files Description](#files-description)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains a hackathon project designed to predict the likelihood of various diseases based on patient medical data. The project utilizes machine learning algorithms to analyze training and testing datasets, providing predictions for:

- **Diabetes** - Predicts diabetes risk based on medical parameters
- **Heart Disease** - Identifies heart disease probability
- **Brain Tumor** - Detects brain tumor presence
- **Kidney Disease** - Assesses kidney disease risk

## 📁 Project Structure

```
healthcare-hackathon/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── SKITECH-INNOTHON-2024.pdf         # Hackathon documentation
│
├── Data Files
├── Training.csv                       # Training dataset
├── Testing.csv                        # Testing dataset
│
├── Model Files (Pre-trained)
├── diabetes.joblib                    # Trained diabetes model
├── heart_save.joblib                  # Trained heart disease model
├── kidney_save.joblib                 # Trained kidney disease model
│
├── Scaler Files
├── scaler.joblib                      # Scaler for diabetes model
├── scaler_heart.joblib                # Scaler for heart model
├── scaler_kidney.joblib               # Scaler for kidney model
│
├── Main Application
├── inputsdisease2.py                  # Main application interface
├── inputsdisease2 copy.py             # Backup of main application
│
├── Prediction Scripts
├── diabitisprediction.py              # Diabetes prediction module
├── heartprediction.py                 # Heart disease prediction module
├── kidney_prediction.py                # Kidney disease prediction module
├── braintumer_prediction.py           # Brain tumor prediction module
│
└── Additional Apps
    ├── app1.py                        # Alternative application
    ├── appher.py                      # Additional prediction interface
    └── appkid.py                      # Kidney disease specific app
```

## ✨ Features

- **Multiple Disease Prediction** - Predict four different diseases from a single interface
- **Pre-trained Models** - Ready-to-use machine learning models
- **Data Scaling** - Proper data normalization using pre-fitted scalers
- **User-Friendly Interface** - Easy-to-use prediction applications
- **Medical Data Analysis** - Based on real medical parameters
- **Modular Design** - Separate prediction modules for each disease

## 🏥 Disease Models

### 1. Diabetes Prediction
- **Model File**: `diabetes.joblib`
- **Scaler**: `scaler.joblib`
- **Script**: `diabitisprediction.py`
- Predicts diabetes risk based on medical health indicators

### 2. Heart Disease Prediction
- **Model File**: `heart_save.joblib`
- **Scaler**: `scaler_heart.joblib`
- **Script**: `heartprediction.py`
- Identifies heart disease probability from cardiac parameters

### 3. Brain Tumor Prediction
- **Model File**: None (model generation script provided)
- **Script**: `braintumer_prediction.py`
- Detects brain tumor presence

### 4. Kidney Disease Prediction
- **Model File**: `kidney_save.joblib`
- **Scaler**: `scaler_kidney.joblib`
- **Scripts**: `kidney_prediction.py`, `appkid.py`
- Assesses kidney disease risk

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sandip7224/healthcare-hackathon.git
   cd healthcare-hackathon
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Requirements

The project requires the following Python packages:
- scikit-learn - Machine learning library
- pandas - Data manipulation
- numpy - Numerical computing
- joblib - Model serialization
- And other dependencies listed in `requirements.txt`

Install all requirements using:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Main Application

```bash
python inputsdisease2.py
```

This launches the main interface where you can:
- Input patient medical parameters
- Select the disease to predict
- Get prediction results

### Running Individual Prediction Modules

**Diabetes Prediction:**
```bash
python diabitisprediction.py
```

**Heart Disease Prediction:**
```bash
python heartprediction.py
```

**Kidney Disease Prediction:**
```bash
python kidney_prediction.py
```

**Brain Tumor Prediction:**
```bash
python braintumer_prediction.py
```

### Alternative Applications

```bash
python app1.py        # Alternative interface
python appher.py      # Additional prediction interface
python appkid.py      # Kidney-specific prediction
```

## 📄 Files Description

| File | Description |
|------|-------------|
| `inputsdisease2.py` | Main application with unified interface for all disease predictions |
| `diabitisprediction.py` | Diabetes prediction implementation |
| `heartprediction.py` | Heart disease prediction implementation |
| `kidney_prediction.py` | Kidney disease prediction implementation |
| `braintumer_prediction.py` | Brain tumor prediction implementation |
| `Training.csv` | Dataset for model training |
| `Testing.csv` | Dataset for model testing and validation |
| `diabetes.joblib` | Serialized trained diabetes model |
| `heart_save.joblib` | Serialized trained heart disease model |
| `kidney_save.joblib` | Serialized trained kidney disease model |
| `scaler*.joblib` | Feature scalers for data normalization |

## 🧠 Model Details

All models are trained using scikit-learn's machine learning algorithms:
- **Algorithm**: Typically classification algorithms (Random Forest, Logistic Regression, etc.)
- **Training Data**: Sourced from medical datasets
- **Accuracy**: Models validated against test datasets
- **Feature Scaling**: Implemented using StandardScaler for normalized input

### Model Loading Example

```python
import joblib

# Load model
model = joblib.load('diabetes.joblib')

# Load scaler
scaler = joblib.load('scaler.joblib')

# Make prediction
prediction = model.predict(scaler.transform(input_data))
```

## 📊 Data Format

Training and testing data should follow the CSV format with medical parameters as features and disease presence as target variables.

## 💡 How It Works

1. **Input**: User provides medical parameters (e.g., glucose level, blood pressure, etc.)
2. **Scaling**: Input data is normalized using the appropriate scaler
3. **Prediction**: Pre-trained model predicts disease probability
4. **Output**: Binary prediction (disease present/absent) or probability score

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add improvement'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Open a Pull Request

## 📝 License

This project is part of the SKITECH INNOTHON 2024 hackathon.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: These models are for educational and demonstration purposes. Always consult with medical professionals for actual disease diagnosis and treatment.
