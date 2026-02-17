Document Classification System using SVM
🔗 Live Demo

👉 https://documentclassification-nxcmce6kxg3nrcvksxceg5.streamlit.app/

📌 Project Overview

This project implements a Document Classification System that automatically classifies text into predefined categories using classical Machine Learning techniques.

The system converts text into numerical form using TF-IDF vectorization and then classifies it using a Support Vector Machine (SVM) model. It also provides a confidence score indicating prediction certainty.

🎯 Objective

To build a lightweight, efficient document classification system that:

runs locally or online

uses classical ML algorithms (no LLMs)

provides probability confidence

handles real-world text data

📂 Dataset

The model was trained using the 20 Newsgroups Dataset from sklearn.datasets.

Dataset Characteristics

~20,000 documents

20 categories

Real-world text data

Standard benchmark dataset for NLP

Each sample consists of:

(Text Document → Category Label)
⚙️ Technologies Used
Component	Tool
Language	Python
ML Library	Scikit-learn
Vectorization	TF-IDF
Model	Linear SVM
Calibration	CalibratedClassifierCV
UI	Streamlit
🧠 Model Architecture

Pipeline:

Text Input
   ↓
TF-IDF Vectorizer
   ↓
SVM Classifier
   ↓
Predicted Category + Confidence Score
🤖 Models Compared

Three classical machine learning models were evaluated:

Model	Role
SVM	Final selected model
Logistic Regression	Comparison model
Naive Bayes	Baseline model
🏆 Why SVM Was Selected

SVM achieved the best performance because:

Handles high-dimensional data well

Works efficiently on sparse text features

Finds optimal decision boundary

Produces strong generalization

📊 Evaluation Metrics

The model was evaluated using:

Accuracy

Precision

Recall

F1 Score

Accuracy Formula:

Accuracy = Correct Predictions / Total Predictions
🔎 Confidence Score

The system outputs a confidence score representing prediction certainty.

Confidence Score = Highest Predicted Probability

SVM normally produces decision scores, so probability calibration was applied using:

CalibratedClassifierCV
🚀 How To Run Locally
1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Run app
streamlit run app.py
📁 Project Structure
project/
│
├── app.py
├── svm_model.pkl
├── requirements.txt
└── README.md
💡 Example Input
NASA launches new satellite

Output:

Category → sci.space
Confidence → 0.93
🎓 Key Learning Outcomes

This project demonstrates:

text preprocessing techniques

feature extraction using TF-IDF

classical machine learning classification

model evaluation

probability calibration

web app deployment

📌 Conclusion

This project shows that classical machine learning algorithms like SVM can achieve high accuracy for text classification tasks when combined with proper preprocessing and feature engineering. The system is lightweight, interpretable, and efficient, making it suitable for real-world deployment.