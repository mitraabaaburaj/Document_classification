import os
import joblib

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score


MODEL_FILE = "svm_model.pkl"

# =========================
# LOAD OR TRAIN MODEL
# =========================
if os.path.exists(MODEL_FILE):

    print("Loading saved model...\n")
    model, vectorizer, class_names = joblib.load(MODEL_FILE)

else:

    print("Training model first time...\n")

    # Load dataset
    data = fetch_20newsgroups(subset='all')
    X, y = data.data, data.target
    class_names = data.target_names

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=50000,
        ngram_range=(1,2),
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    base = LinearSVC(class_weight='balanced', C=1.5, dual='auto')
    model = CalibratedClassifierCV(base)

    print("Training SVM...")
    model.fit(X_train_vec, y_train)

    # Evaluation
    preds = model.predict(X_test_vec)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=class_names))

    # Save model
    joblib.dump((model, vectorizer, class_names), MODEL_FILE)
    print("\nModel saved as", MODEL_FILE)


# =========================
# EXAMPLE PREDICTIONS
# =========================
print("\n===== EXAMPLE PREDICTIONS =====")

examples = [
    "NASA launches new satellite",
    "New graphics card released",
    "Government passes new law",
    "Baseball match was exciting"
]

for text in examples:
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    conf = max(model.predict_proba(vec)[0])

    print(f"\nText: {text}")
    print(f"Class → {pred} ({class_names[pred]})")
    print("Confidence:", round(conf, 4))


# =========================
# USER INPUT LOOP
# =========================
while True:
    text = input("\nEnter text (exit to quit): ")

    if text.lower() == "exit":
        break

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    conf = max(model.predict_proba(vec)[0])

    print(f"Class → {pred} ({class_names[pred]})")
    print("Confidence:", round(conf, 4))
