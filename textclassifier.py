# ── Practical 9: Text Classifier using Keras (with IMDB dataset) ──
#download the imdb dataset for this from https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── 1. Load IMDB dataset (same one from Practical 8)
df = pd.read_csv('IMDB Dataset.csv')
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Use 10,000 samples to keep training fast
df = df.sample(10000, random_state=42).reset_index(drop=True)
texts  = df['review'].tolist()
labels = df['label'].tolist()

print(f"Dataset size : {len(texts)}")
print(f"Positive     : {sum(labels)}")
print(f"Negative     : {len(labels) - sum(labels)}")

# ── 2. Tokenize & Pad
MAX_WORDS = 10000
MAX_LEN   = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded    = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# ── 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    padded, np.array(labels), test_size=0.2, random_state=42)

# ── 4. Build Model
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    LSTM(64, return_sequences=True),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ── 5. Train
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# ── 6. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.2%}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

# ── 7. Predict on new text
def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = model.predict(pad, verbose=0)[0][0]
    label = "POSITIVE" if prob > 0.5 else "NEGATIVE"
    print(f"  '{text}'\n  → {label} (confidence: {prob:.2%})\n")

print("\n── Custom Predictions ──")
predict("This is the best thing I ever bought!")
predict("Absolutely terrible, never again.")
predict("Decent product, nothing special.")
predict("Waste of my money")
predict("Could have bought something more better with the money i spent on this")
