import streamlit as st
import joblib
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Document Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Document Classification System")
st.write("Classifies text into categories using SVM")

# ===============================
# LOAD MODEL
# ===============================
MODEL_FILE = "svm_model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("Model file not found. Run training script first.")
    st.stop()

model, vectorizer, class_names = joblib.load(MODEL_FILE)

st.success("Model loaded successfully!")

# ===============================
# USER INPUT
# ===============================
st.subheader("Enter Text")

user_text = st.text_area(
    "Type a sentence or paragraph",
    height=150,
    placeholder="Example: NASA launches new rocket"
)

# ===============================
# PREDICT BUTTON
# ===============================
if st.button("Classify Document"):

    if user_text.strip() == "":
        st.warning("Please enter text.")
    else:
        vec = vectorizer.transform([user_text])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = max(probs)

        st.markdown("### 📊 Prediction Result")
        st.write(f"**Category:** {class_names[pred]}")
        st.write(f"**Confidence:** {round(confidence,4)}")

        # show all probabilities
        st.markdown("### 🔎 Class Probabilities")
        for i, p in enumerate(probs):
            st.write(f"{class_names[i]} → {round(p,4)}")

# ===============================
# EXAMPLE TEXTS
# ===============================
st.markdown("---")
st.subheader("Try Example Inputs")

examples = [
    "NASA launches satellite into orbit",
    "The baseball match was thrilling",
    "New graphics card released today",
    "Government announces new policy"
]

for ex in examples:
    if st.button(ex):
        vec = vectorizer.transform([ex])
        pred = model.predict(vec)[0]
        confidence = max(model.predict_proba(vec)[0])

        st.write("Prediction:", class_names[pred])
        st.write("Confidence:", round(confidence,4))
