from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data=fetch_20newsgroups(subset='all')
X,y=data.data,data.target
class_names=data.target_names

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

vectorizer=TfidfVectorizer(stop_words='english',max_features=50000,ngram_range=(1,2),sublinear_tf=True)
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)

model=LogisticRegression(max_iter=2000,class_weight='balanced',n_jobs=-1)

print("Training Logistic Regression...")
model.fit(X_train_vec,y_train)

preds=model.predict(X_test_vec)
print("\nAccuracy:",accuracy_score(y_test,preds))
print(classification_report(y_test,preds,target_names=class_names))


examples=["New medical discovery announced","New car engine released","Football match tonight"]

print("\n===== EXAMPLES =====")
for t in examples:
    vec=vectorizer.transform([t])
    p=model.predict(vec)[0]
    conf=max(model.predict_proba(vec)[0])
    print(f"\nText: {t}")
    print(f"Class → {p} ({class_names[p]})")
    print("Confidence:",round(conf,4))


while True:
    t=input("\nEnter text (exit): ")
    if t.lower()=="exit": break
    vec=vectorizer.transform([t])
    p=model.predict(vec)[0]
    conf=max(model.predict_proba(vec)[0])
    print(f"Class → {p} ({class_names[p]})")
    print("Confidence:",round(conf,4))
