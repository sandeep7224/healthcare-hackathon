#interactive chatbot 

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import warnings

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

# Function to find the closest matching symptom
def get_closest_symptom(input_symptom, feature_list):
    symptom, score = process.extractOne(input_symptom, feature_list)
    return symptom if score >= 80 else None

# Function to predict disease based on user symptoms
def predict_disease(symptoms_list):
    input_data = np.zeros(len(features))  # Create an input array with all features initialized to 0
    
    for symptom in symptoms_list:
        if symptom in features:
            index = list(features).index(symptom)
            input_data[index] = 1

    input_data = np.array([input_data])  # Reshape for the model input
    prediction_encoded = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)
    return prediction[0]

# Recommendations based on diseases
# Recommendations for each disease
disease_recommendations = {
    'Fungal infection': {
        'precautions': ["Use antifungal creams or ointments", "Keep the area dry and clean", "Avoid tight-fitting clothing"],
        'medications': ["Topical antifungal creams", "Oral antifungal medications if needed"],
        'exercises': ["Gentle stretching exercises", "Avoid activities that cause excessive sweating"]
    },
    'Allergy': {
        'precautions': ["Avoid allergens", "Use air purifiers", "Keep windows closed during pollen season"],
        'medications': ["Antihistamines", "Nasal corticosteroids", "Decongestants"],
        'exercises': ["Breathing exercises", "Gentle yoga to relieve congestion"]
    },
    'GERD': {
        'precautions': ["Avoid spicy and acidic foods", "Eat smaller meals", "Avoid lying down after eating"],
        'medications': ["Antacids", "Proton pump inhibitors", "H2 blockers"],
        'exercises': ["Avoid high-intensity workouts", "Gentle walking post meals"]
    },
    'Chronic cholestasis': {
        'precautions': ["Avoid alcohol", "Maintain a healthy weight", "Follow a low-fat diet"],
        'medications': ["Ursodeoxycholic acid", "Vitamin supplements"],
        'exercises': ["Moderate-intensity exercises", "Avoid strenuous activities"]
    },
    'Drug Reaction': {
        'precautions': ["Avoid known allergens", "Read medication labels carefully", "Consult a doctor before taking new medications"],
        'medications': ["Antihistamines", "Steroids"],
        'exercises': ["Light exercises to stay active", "Avoid strenuous activities until reaction subsides"]
    },
    'Peptic ulcer disease': {
        'precautions': ["Avoid spicy foods", "Limit alcohol and caffeine", "Quit smoking"],
        'medications': ["Antacids", "Proton pump inhibitors", "H2 blockers"],
        'exercises': ["Gentle yoga", "Walking to improve digestion"]
    },
    'AIDS': {
        'precautions': ["Practice safe sex", "Avoid sharing needles", "Take antiretroviral therapy as prescribed"],
        'medications': ["Antiretroviral therapy (ART)", "Medications for opportunistic infections"],
        'exercises': ["Regular aerobic exercises", "Strength training"]
    },
    'Diabetes': {
        'precautions': ["Monitor blood sugar levels", "Follow a balanced diet", "Exercise regularly"],
        'medications': ["Insulin", "Oral hypoglycemic agents"],
        'exercises': ["Aerobic exercises like walking or swimming", "Strength training"]
    },
    'Gastroenteritis': {
        'precautions': ["Stay hydrated", "Eat small, bland meals", "Avoid dairy and caffeine"],
        'medications': ["Oral rehydration solutions", "Anti-diarrheal medications"],
        'exercises': ["Rest until symptoms improve", "Gentle stretching"]
    },
    'Bronchial Asthma': {
        'precautions': ["Avoid allergens", "Use a humidifier", "Take prescribed medications regularly"],
        'medications': ["Bronchodilators", "Inhaled corticosteroids"],
        'exercises': ["Breathing exercises", "Low-intensity aerobic exercises"]
    },
    'Hypertension': {
        'precautions': ["Monitor blood pressure regularly", "Limit salt intake", "Maintain a healthy weight"],
        'medications': ["ACE inhibitors", "Beta blockers", "Diuretics"],
        'exercises': ["Aerobic exercises", "Yoga and relaxation exercises"]
    },
    'Migraine': {
        'precautions': ["Avoid known triggers", "Practice stress management", "Maintain a regular sleep schedule"],
        'medications': ["Pain relievers", "Triptans", "Anti-nausea medications"],
        'exercises': ["Gentle yoga", "Breathing exercises"]
    },
    'Cervical spondylosis': {
        'precautions': ["Maintain good posture", "Avoid heavy lifting", "Use a supportive pillow"],
        'medications': ["Pain relievers", "Muscle relaxants"],
        'exercises': ["Neck stretches", "Strengthening exercises for neck and shoulders"]
    },
    'Paralysis (brain hemorrhage)': {
        'precautions': ["Follow a rehabilitation program", "Prevent falls and injuries", "Monitor blood pressure"],
        'medications': ["Anticoagulants", "Medications to control blood pressure"],
        'exercises': ["Physical therapy", "Strength training"]
    },
    'Jaundice': {
        'precautions': ["Avoid alcohol", "Follow a low-fat diet", "Stay hydrated"],
        'medications': ["Medications to treat underlying cause", "Supportive care"],
        'exercises': ["Gentle walking", "Avoid strenuous activities until recovery"]
    },
    'Malaria': {
        'precautions': ["Use mosquito nets", "Apply insect repellent", "Take antimalarial drugs when traveling to high-risk areas"],
        'medications': ["Antimalarial medications", "Pain relievers"],
        'exercises': ["Rest until symptoms improve", "Light exercises once recovered"]
    },
    'Chicken pox': {
        'precautions': ["Avoid scratching", "Keep the skin clean", "Stay hydrated"],
        'medications': ["Antihistamines", "Antiviral drugs"],
        'exercises': ["Rest until fever subsides", "Light stretching"]
    },
    'Dengue': {
        'precautions': ["Use mosquito repellent", "Wear long-sleeved clothing", "Avoid mosquito bites"],
        'medications': ["Pain relievers", "Fluids for hydration"],
        'exercises': ["Rest", "Avoid physical exertion until fully recovered"]
    },
    'Typhoid': {
        'precautions': ["Drink clean water", "Wash hands regularly", "Avoid raw foods"],
        'medications': ["Antibiotics", "Pain relievers"],
        'exercises': ["Rest", "Gentle walking once symptoms improve"]
    },
    'Hepatitis B': {
        'precautions': ["Practice safe sex", "Avoid sharing needles", "Get vaccinated"],
        'medications': ["Antiviral medications", "Liver supportive care"],
        'exercises': ["Moderate-intensity exercises", "Avoid strenuous activities"]
    },
    'Hepatitis C': {
        'precautions': ["Avoid sharing personal items", "Get vaccinated for hepatitis A and B", "Practice safe sex"],
        'medications': ["Antiviral medications", "Liver supportive care"],
        'exercises': ["Moderate-intensity exercises", "Avoid strenuous activities"]
    },
    'Hepatitis D': {
        'precautions': ["Avoid sharing needles", "Practice safe sex", "Get vaccinated for hepatitis B"],
        'medications': ["Antiviral medications", "Liver supportive care"],
        'exercises': ["Moderate-intensity exercises", "Avoid strenuous activities"]
    },
    'Hepatitis E': {
        'precautions': ["Drink clean water", "Maintain good hygiene", "Avoid undercooked meat"],
        'medications': ["Supportive care", "Rest"],
        'exercises': ["Light activities", "Rest until recovery"]
    },
    'Alcoholic hepatitis': {
        'precautions': ["Stop alcohol consumption", "Follow a healthy diet", "Regularly monitor liver function"],
        'medications': ["Corticosteroids", "Liver supportive care"],
        'exercises': ["Avoid strenuous activities", "Gentle walking"]
    },
    'Tuberculosis': {
        'precautions': ["Take medications regularly", "Cover mouth when coughing", "Stay in well-ventilated areas"],
        'medications': ["Antibiotics", "Supportive care"],
        'exercises': ["Breathing exercises", "Low-intensity physical activity"]
    },
    'Common Cold': {
        'precautions': ["Wash hands regularly", "Avoid close contact with sick individuals", "Stay warm"],
        'medications': ["Decongestants", "Cough suppressants"],
        'exercises': ["Rest", "Light walking"]
    },
    'Pneumonia': {
        'precautions': ["Get vaccinated", "Avoid smoking", "Wash hands regularly"],
        'medications': ["Antibiotics", "Cough medicine"],
        'exercises': ["Breathing exercises", "Rest until recovery"]
    },
    'Dimorphic hemmorhoids(piles)': {
        'precautions': ["Eat a high-fiber diet", "Stay hydrated", "Avoid straining during bowel movements"],
        'medications': ["Topical treatments", "Pain relievers"],
        'exercises': ["Kegel exercises", "Walking to improve circulation"]
    },
    'Heart attack': {
        'precautions': ["Avoid smoking", "Maintain a healthy diet", "Exercise regularly"],
        'medications': ["Aspirin", "Beta-blockers", "ACE inhibitors"],
        'exercises': ["Cardiac rehabilitation", "Moderate-intensity exercises"]
    },
    'Varicose veins': {
        'precautions': ["Avoid prolonged standing", "Wear compression stockings", "Elevate legs"],
        'medications': ["Pain relievers", "Sclerotherapy"],
        'exercises': ["Leg exercises", "Walking to improve circulation"]
    },
    'Hypothyroidism': {
        'precautions': ["Take thyroid hormone replacement", "Eat a balanced diet", "Exercise regularly"],
        'medications': ["Levothyroxine", "Thyroid hormone replacement"],
        'exercises': ["Aerobic exercises", "Strength training"]
    },
    'Hyperthyroidism': {
        'precautions': ["Avoid iodine-rich foods", "Take medications as prescribed", "Manage stress"],
        'medications': ["Antithyroid medications", "Beta-blockers"],
        'exercises': ["Low-impact exercises", "Yoga and relaxation techniques"]
    },
    'Hypoglycemia': {
        'precautions': ["Eat small, frequent meals", "Avoid skipping meals", "Monitor blood sugar levels"],
        'medications': ["Glucose tablets", "Emergency glucagon kit"],
        'exercises': ["Moderate-intensity exercises", "Walking"]
    },
    'Osteoarthritis': {
        'precautions': ["Maintain a healthy weight", "Exercise regularly", "Use supportive devices if needed"],
        'medications': ["Pain relievers", "Anti-inflammatory drugs"],
        'exercises': ["Joint-friendly exercises", "Strengthening exercises"]
    },
    '(Vertigo) Paroxysmal Positional Vertigo': {
        'precautions': ["Avoid sudden head movements", "Stay hydrated", "Sleep with head elevated"],
        'medications': ["Antivertigo medications", "Antihistamines"],
        'exercises': ["Vestibular rehabilitation exercises", "Balance exercises"]
    },
    'Acne': {
        'precautions': ["Clean skin regularly", "Avoid picking at pimples", "Use non-comedogenic products"],
        'medications': ["Topical treatments", "Oral antibiotics"],
        'exercises': ["Avoid excessive sweating", "Maintain good hygiene"]
    },
    'Urinary tract infection': {
        'precautions': ["Drink plenty of water", "Urinate after sexual activity", "Avoid irritants"],
        'medications': ["Antibiotics", "Pain relievers"],
        'exercises': ["Kegel exercises", "Pelvic floor strengthening"]
    },
    'Psoriasis': {
        'precautions': ["Moisturize regularly", "Avoid triggers", "Manage stress"],
        'medications': ["Topical treatments", "Phototherapy"],
        'exercises': ["Gentle stretching", "Low-impact exercises"]
    },
    'Impetigo': {
        'precautions': ["Keep skin clean", "Avoid scratching", "Cover affected areas"],
        'medications': ["Antibiotic ointments", "Oral antibiotics"],
        'exercises': ["Avoid physical contact sports", "Light activities"]
    },
    'Arthritis': {
        'precautions': ["Stay active", "Maintain a healthy weight", "Use assistive devices if needed"],
        'medications': ["Pain relievers", "Anti-inflammatory drugs"],
        'exercises': ["Joint-friendly exercises", "Range-of-motion exercises"]
    },
    'Vomiting': {
        'precautions': ["Stay hydrated", "Avoid solid foods until vomiting stops", "Eat bland foods"],
        'medications': ["Antiemetic medications", "Hydration solutions"],
        'exercises': ["Rest", "Avoid physical exertion"]
    },
    'Hepatitis E': {
        'precautions': ["Maintain good hygiene", "Avoid undercooked meat", "Drink clean water"],
        'medications': ["Supportive care", "Rest"],
        'exercises': ["Light activities", "Rest until recovery"]
    }
}


# Function to ask follow-up questions based on provided symptoms
def ask_followup_question(symptoms):
    # Extended follow-up questions based on common symptom categories
    symptom_categories = {
        'fever': ["Do you have chills?", "Is your fever high or low grade?", "Do you have night sweats?"],
        'cough': ["Is it a dry cough or with phlegm?", "Do you have chest pain?", "Are you coughing up blood?"],
        'pain': ["Where do you feel the pain?", "Is the pain sharp or dull?", "Does the pain come and go?"],
        'fatigue': ["Do you feel tired even after resting?", "How long have you been feeling fatigued?", "Do you have trouble sleeping?"],
        'headache': ["Is the headache mild or severe?", "Do you feel any nausea with the headache?", "Does light or noise make it worse?"],
        'sore throat': ["Do you have difficulty swallowing?", "Is your throat scratchy or painful?", "Do you have swollen glands?"],
        'nausea': ["Have you been vomiting?", "Do you feel nauseous after eating?", "Do you have any stomach pain?"],
        'diarrhea': ["How many times have you had diarrhea today?", "Is there any blood in your stool?", "Do you have stomach cramps?"],
        'rash': ["Where is the rash located?", "Is the rash itchy or painful?", "Have you had this rash before?"],
        'dizziness': ["Do you feel lightheaded when standing?", "Have you fainted recently?", "Do you have balance problems?"]
    }
    
    for symptom in symptoms:
        for category, questions in symptom_categories.items():
            if category in symptom:
                return questions[0]  # Ask the first follow-up question found for this category
    return None

# Start the interactive session with the user
print("Hi! I'm here to help you diagnose your symptoms. Let's start.")

user_symptoms = []
while True:
    # Start with a general question or ask a follow-up question based on the previous input
    if user_symptoms:
        followup = ask_followup_question(user_symptoms)
        if followup:
            print(followup)
    symptom = input("Describe a symptom you are experiencing: ").strip().lower()
    
    if symptom == 'stop':
        break

    closest_symptom = get_closest_symptom(symptom, features)
    
    if closest_symptom:
        print(f"Did you mean: {closest_symptom}?")
        user_symptoms.append(closest_symptom)
    else:
        print("Symptom not recognized or too dissimilar. Please try again.")

    # Ask if the user wants to continue or stop
    continue_input = input("Would you like to enter more symptoms? (yes to continue, 'stop' to end): ").strip().lower()
    if continue_input == 'stop':
        break

if user_symptoms:
    predicted_disease = predict_disease(user_symptoms)
    print(f"Based on the symptoms you provided, the predicted disease is: {predicted_disease}")
    
    if predicted_disease in disease_recommendations:
        print("Here are some recommendations for you:")
        recommendations = disease_recommendations[predicted_disease]
        for key, value in recommendations.items():
            print(f"{key}: {value}")
else:
    print("No symptoms provided. Unable to make a prediction.")
