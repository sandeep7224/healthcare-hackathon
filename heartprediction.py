import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv('heart.csv')  # Replace with your dataset filename

# Separate features and target variable
X = df.drop('target', axis=1)  # 'target' is the name of the target column
y = df['target']  # 'target' is the name of the target column

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)



joblib.dump(model_rf,"heart_save.joblib")
joblib.dump(scaler, 'scaler_heart.joblib')