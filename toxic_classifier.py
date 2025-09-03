import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("train.csv")[['comment_text', 'toxic']]  # Use only 'toxic' label
df = df.dropna()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['toxic'], test_size=0.2, random_state=42)

# Text preprocessing + TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# Predict
y_pred = clf.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Function for predictions
def predict_toxicity(text):
    vec = tfidf.transform([text])
    pred = clf.predict(vec)[0]
    return "Toxic" if pred == 1 else "Not Toxic"

# Example tests
print(predict_toxicity("You are so stupid!"))
print(predict_toxicity("I love this!"))
