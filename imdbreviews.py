# ── Practical 8: IMDB Sentiment Classification ────────────────────
# Download dataset from Kaggle first, then upload to Colab
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (adjust path if needed)
df = pd.read_csv('IMDB Dataset.csv')
print(df.head(2))
print("Shape:", df.shape)

# Encode labels
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42)

# TF-IDF features
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

custom = ["This movie was absolutely brilliant!", "Terrible plot, waste of time.", "why did I waste my money on this", "This movie is so bad  brotherrr."]
vecs = tfidf.transform(custom)
preds = clf.predict(vecs)
for review, pred in zip(custom, preds):
    print(f"  {'POSITIVE' if pred else 'NEGATIVE'}: {review}")
