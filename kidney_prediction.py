# Import necessary libraries
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset, ensuring that unnecessary columns are not loaded
df = pd.read_csv('kidneydisease.csv')  # Replace with your actual file path
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Separate features and target variable
X = df.drop(['id', 'classification'], axis=1)  # Assuming 'id' is not a feature and 'classification' is the target
y = df['classification']

# Standardize features (important for many machine learning algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)



joblib.dump(rf_model,"kidney_save.joblib")
joblib.dump(scaler, 'scaler_kidney.joblib')