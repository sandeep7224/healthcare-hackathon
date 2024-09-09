import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import warnings
import joblib

warnings.filterwarnings("ignore")

# Load your training and testing datasets
train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing.csv')

# Assuming the last column is 'Disease' (the label)
features = train_df.columns[:-1]  # All columns except the last one
label_column = 'Disease'

# Fill missing values if any
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# Convert categorical symptom columns to numerical data
X_train = train_df[features]
y_train = train_df[label_column]
X_test = test_df[features]
y_test = test_df[label_column]

# Encode the disease labels into numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Ensure X_train and X_test are numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train_encoded)

joblib.dump(model, 'main_text.joblib')

