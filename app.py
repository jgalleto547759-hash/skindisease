# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Symptom Checker (Educational Only)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Cache heavy objects (model, embedder, etc.)
# ──────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_model_and_le(_embedder):
    # ── Your original data loading & training logic ──
    dataset_path = "Symptom2Disease.csv"

    if not os.path.exists(dataset_path):
        st.warning("Downloading dataset...")
        os.system(f"wget -q https://raw.githubusercontent.com/jatinmandav/Disease-Prediction-from-Symptoms/main/Symptom2Disease.csv -O {dataset_path}")

    df = pd.read_csv(dataset_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_cat = tf.keras.utils.to_categorical(y_encoded)
    num_classes = y_cat.shape[1]

    X_emb = _embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_emb, y_cat,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    inputs = layers.Input(shape=(384,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # We won't actually re-train every time → load if exists, else train once
    model_path = "symptom_model.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        with st.spinner("Training model (first run only — takes ~1–2 min)..."):
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=80,
                batch_size=16,
                callbacks=callbacks,
                verbose=0
            )
            model.save(model_path)
            st.success("Model trained and saved!")

    return model, le

# ──────────────────────────────────────────────
# Responses dictionary (your original)
# ──────────────────────────────────────────────
responses = {
    "Psoriasis": "It looks like possible **Psoriasis** — immune system speeds up skin cell growth → red, scaly, itchy patches (elbows, knees, scalp common). Humidity & stress in Davao can worsen it.\n\nTry thick fragrance-free moisturizer 3–4× daily, lukewarm showers, avoid harsh soap/scratching.\nImprovement often in 5–10 days.\nSee dermatologist if spreading fast, cracking/bleeding or no change after ~2 weeks.",

    "diabetes": "Thirst, frequent urination, tiredness, weight loss, sometimes blurry vision — possible **Diabetes**.\n\nCut sugar, balanced meals, water.\n**See doctor soon** — blood sugar/HbA1c test needed.",

    "Varicose Veins": "This matches **Varicose Veins** — veins stretch & twist (usually legs) from poor blood return. Standing long, heat & humidity make it worse here.\n\nElevate legs, walk regularly, try compression stockings, avoid tight clothes.\nIf aching improves in 1–2 weeks great; see doctor if veins worsen, skin darkens or ulcers appear.",

    "Typhoid": "Symptoms raise concern for **Typhoid fever** — bacterial infection from contaminated food/water.\n\nRest, sip ORS, eat light food.\n**Do not wait** — see doctor/hospital today for blood test & antibiotics. Can become serious.",

    "Chicken pox": "Typical **Chickenpox** — itchy rash → blisters, fever, tiredness. Very contagious.\n\nCalamine lotion, cool baths, trim nails, rest, fluids.\nUsually clears 7–14 days. Doctor immediately if high fever returns, pus/red streaks, breathing issues or pregnancy.",

    "Impetigo": "Looks like **Impetigo** — contagious bacterial sores with honey crusts (mouth/nose/limbs common). Warm humid weather spreads it.\n\nClean gently, keep dry.\nDoctor if spreading, fever or no improvement in 3–5 days (needs antibiotics).",

    "Dengue": "High fever, severe aches, headache, possible rash — very suggestive of **Dengue** (mosquito virus, rainy season risk).\n\nRest, lots of fluids (ORS), paracetamol only.\n**Urgent hospital** if severe pain, vomiting, bleeding, weakness or fever returns after drop.",

    "Fungal infection": "Likely **Fungal infection** (ringworm etc.) — red scaly itchy patches, loves warm moist skin (Davao common).\n\nKeep dry, loose cotton, OTC antifungal cream 2× daily.\nImproves 5–10 days; dermatologist if spreading or no change after 2–3 weeks.",

    "Common Cold": "Most likely **Common Cold** (viral) — runny nose, sore throat, mild cough, low fever.\n\nRest, fluids, honey/lemon, saline spray. Better in 7–10 days.\nDoctor if high fever >3 days, breathing trouble or worsening.",

    "Pneumonia": "Cough/phlegm, fever, chills, chest pain, shortness of breath — possible **Pneumonia**.\n\nRest, hydrate, avoid smoke.\n**See doctor soon** (today if hard breathing/high fever/confusion) — may need antibiotics/hospital.",

    "Dimorphic Hemorrhoids": "Pain/itching/bleeding on bowel movement — classic **Hemorrhoids**.\n\nMore fiber (fruits/veggies/oats), plenty water, avoid straining, warm sitz baths.\nDoctor if bleeding continues or severe pain after 5–7 days.",

    "Arthritis": "Joint pain/stiffness/swelling — likely **Arthritis**.\n\nGentle movement, warm/cold packs, healthy weight.\nDoctor if hot/red joints, can't move or >2 weeks.",

    "Acne": "Blackheads/whiteheads/pimples/cysts — typical **Acne**.\n\nGentle wash 2× day, non-comedogenic moisturizer, avoid picking.\nDermatologist if cysts/scarring or no improvement 6–8 weeks.",

    "Bronchial Asthma": "Wheezing, shortness of breath, chest tightness, night cough — sounds like **Asthma**.\n\nAvoid triggers, use inhaler if prescribed.\n**Emergency** if can't speak full sentences or blue lips.",

    "Hypertension": "Headache/dizziness/blurred vision — possible **High Blood Pressure**.\n\nCut salt, walk 30 min/day, manage stress, check BP.\nDoctor urgently if >140/90 or severe symptoms.",

    "Migraine": "Throbbing headache, nausea, light/sound sensitivity — classic **Migraine**.\n\nDark room, hydrate, cold compress, avoid triggers.\nDoctor if >4/month, >72 h or new symptoms.",

    "Cervical spondylosis": "Neck pain/stiffness, arm/shoulder pain, numbness — **Cervical spondylosis**.\n\nGood posture, gentle stretches, heat pack.\nDoctor/PT if severe/weakness/balance issues.",

    "Jaundice": "Yellow skin/eyes, dark urine, fatigue — **Jaundice** (liver issue).\n\nNo alcohol, rest, hydrate.\n**See doctor today** — needs blood tests.",

    "Malaria": "Cyclic fever/chills/sweating, headache, aches — suggestive **Malaria**.\n\nRest, hydrate, paracetamol.\n**Hospital immediately** — test & treatment needed.",

    "urinary tract infection": "Burning pee, frequent urge, lower pain — classic **UTI**.\n\nDrink 2–3 L water, urinate often.\nDoctor in 1–2 days if no better or fever/back pain (antibiotics).",

    "allergy": "Itching/rash/hives/sneezing — likely **Allergy**.\n\nAvoid trigger, antihistamine, cool compress.\n**Emergency** if swelling lips/throat or breathing trouble.",

    "gastroesophageal reflux disease": "Heartburn/sour taste, worse after meals/lying — **GERD**.\n\nSmall meals, avoid spicy/acidic, don't lie down after eating.\nDoctor if >2×/week or swallowing trouble.",

    "drug reaction": "New rash/itching after medicine — possible **Drug Reaction**.\n\nStop suspected drug (check with pharmacist), hydrate.\nUrgent if spreading fast/blisters/fever.",

    "peptic ulcer disease": "Burning stomach pain, bloating, nausea — **Peptic Ulcer**.\n\nAvoid NSAIDs/spicy/alcohol, small meals, antacids temp relief.\nDoctor — may need PPI + H. pylori test."
}


default_response = """I'm not confident yet. Could you describe your symptoms more clearly?

Helpful details:
• Where is it exactly?
• How long?
• Painful/itchy/burning/numb?
• Fever/thirst/tiredness/swelling?
• Worse after work/standing/heat?

More info helps — but please see a doctor for real concerns!"""

blurry_exertion_reply = """**⚠️ NOT a diagnosis – educational only! See a doctor.**

Blurry vision during/after hard work often relates to **blood sugar changes** ...

(keep your full text)
"""

# ──────────────────────────────────────────────
# Prediction functions
# ──────────────────────────────────────────────
def predict_disease(text, embedder, model, le):
    emb = embedder.encode([text], convert_to_numpy=True)
    pred = model.predict(emb, verbose=0)
    idx = np.argmax(pred)
    disease = le.inverse_transform([idx])[0]
    conf = pred[0][idx] * 100
    return disease, conf

def get_bot_reply(user_input, embedder, model, le):
    input_lower = user_input.lower()

    if any(w in input_lower for w in ["blurry", "blur", "vision"]) and \
       any(w in input_lower for w in ["work", "hard", "exercise", "effort", "after", "during"]):
        return blurry_exertion_reply

    if len(user_input.split()) < 6:
        return default_response

    disease, conf = predict_disease(user_input, embedder, model, le)

    if conf < 70:
        return f"Low confidence ({conf:.1f}%). Could you add more details?\n\n{default_response}"

    reply = responses.get(disease, default_response)

    return (
        "**⚠️ IMPORTANT: This is NOT a medical diagnosis!**\n"
        "**Educational guess only — consult a real doctor.**\n\n"
        f"{reply}\n"
    )

# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
st.title("🩺 Symptom Checker")
st.caption("Educational tool only – **never** a substitute for professional medical advice")

# Load resources (cached)
embedder = load_embedder()
model, le = load_model_and_le(embedder)

# ── Chat history ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Describe your symptoms (be as detailed as possible)..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_bot_reply(prompt, embedder, model, le)
        st.markdown(response)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

# ── Sidebar extras ──
with st.sidebar:
    st.header("About")
    st.markdown("""
    This is an **educational demonstration** using:
    - Sentence-BERT embeddings
    - Simple feed-forward neural network
    - Symptom → Disease dataset (~1200 examples)

    **Always consult a qualified doctor** for health concerns.
    """)

    if st.button("Clear chat history", type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Deployed with Streamlit • Model saved locally after first run")
