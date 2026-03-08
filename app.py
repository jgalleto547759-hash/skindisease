import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


st.title("Skin Disease Prediction Chatbot")
st.write("Enter your symptoms to predict possible skin diseases.")

symptoms = st.text_input("Enter symptoms (comma separated):")


data = pd.read_csv("Symptom2Disease.csv")

X_text = data["text"]
y = data["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5, batch_size=8, verbose=0)


disease_advice = {
    "Psoriasis": "It seems like you may have Psoriasis. Keep your skin moisturized, avoid harsh soaps, and consult a dermatologist if it worsens.",
    "Varicose Veins": "This looks like Varicose Veins. Avoid standing for long periods, elevate your legs, and consider medical evaluation for compression therapy.",
    "Typhoid": "Symptoms suggest Typhoid. Stay hydrated, rest, and consult a doctor for antibiotics.",
    "Chicken pox": "These symptoms are consistent with Chickenpox. Keep the rash clean, avoid scratching, and isolate from others.",
    "Impetigo": "It may be Impetigo. Keep affected areas clean, avoid scratching, and see a doctor for antibiotics.",
    "Dengue": "Symptoms may indicate Dengue. Stay hydrated, rest, and seek medical attention if fever is high or bleeding occurs.",
    "Fungal infection": "It seems like a Fungal infection. Keep the area dry, use antifungal creams, and consult a dermatologist if it doesn’t improve.",
    "Common Cold": "This may be a Common Cold. Rest, stay hydrated, and consider OTC cold medicines.",
    "Pneumonia": "Symptoms may indicate Pneumonia. See a doctor for proper diagnosis and treatment.",
    "Dimorphic Hemorrhoids": "These symptoms suggest Dimorphic Hemorrhoids. Avoid straining, increase fiber intake, and consult a doctor.",
    "Arthritis": "It could be Arthritis. Maintain joint mobility, use pain relief if needed, and see a doctor.",
    "Acne": "Mild: Keep your skin clean, wash gently twice a day, avoid touching or picking at your face, and use OTC acne creams. Severe: Avoid picking, maintain a gentle routine, and consult a dermatologist for treatment.",
    "Bronchial Asthma": "It may be Bronchial Asthma. Avoid triggers, use prescribed inhalers, and seek medical guidance if symptoms worsen.",
    "Hypertension": "It seems like Hypertension. Monitor blood pressure regularly, maintain a healthy diet, and consult a doctor.",
    "Migraine": "Symptoms may indicate Migraine. Rest in a dark room, manage stress, and consult a doctor for medication.",
    "Cervical spondylosis": "It may be Cervical spondylosis. Maintain good posture, do gentle exercises, and consult a physician.",
    "Jaundice": "These symptoms suggest Jaundice. Avoid alcohol, maintain hydration, and see a doctor.",
    "Malaria": "It could be Malaria. Seek immediate medical attention for testing and treatment.",
    "urinary tract infection": "Symptoms suggest a UTI. Drink plenty of water, maintain hygiene, and see a doctor for antibiotics.",
    "allergy": "It may be an Allergy. Identify and avoid triggers, and consider antihistamines or a doctor consultation if severe.",
    "gastroesophageal reflux disease": "These symptoms may indicate GERD. Avoid spicy foods, eat smaller meals, and consult a doctor.",
    "drug reaction": "It looks like a Drug Reaction. Stop the suspected medication and see a doctor immediately.",
    "peptic ulcer disease": "These symptoms suggest Peptic Ulcer Disease. Avoid NSAIDs, eat light meals, and consult a doctor.",
    "diabetes": "It may be Diabetes. Monitor blood sugar, maintain a healthy diet, and consult a doctor for management."
}


def chatbot_reply(text):
    vec = vectorizer.transform([text]).toarray()
    pred = model.predict(vec)
    label = le.inverse_transform([np.argmax(pred)])[0]
    advice = disease_advice.get(label, "Monitor symptoms and take care.")
    return label, advice


if st.button("Predict"):

    if symptoms.strip() == "":
        st.warning("Please enter symptoms.")
    else:
        disease, advice = chatbot_reply(symptoms)

        st.success(f"Predicted Disease: {disease}")
        st.write(advice)