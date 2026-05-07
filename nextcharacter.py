# ── Practical 10: Character-level Language Model ──────────────────
import numpy as np, random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import LambdaCallback

# Use a Shakespeare excerpt (or any plain text)
#upload the shakespeare.txt file in this repo to colab
text = open('shakespeare.txt').read().lower()   # upload your file
# Quick test: text = "to be or not to be that is the question " * 50

chars     = sorted(set(text))
char2idx  = {c: i for i, c in enumerate(chars)}
idx2char  = {i: c for c, i in char2idx.items()}
print(f"Corpus: {len(text)} chars | Vocab: {len(chars)}")

SEQ_LEN = 40; STEP = 3
X, y = [], []
for i in range(0, len(text) - SEQ_LEN, STEP):
    X.append([char2idx[c] for c in text[i:i+SEQ_LEN]])
    y.append(char2idx[text[i+SEQ_LEN]])
X = tf.keras.utils.to_categorical(X, num_classes=len(chars))
y = tf.keras.utils.to_categorical(y, num_classes=len(chars))

# LSTM model
model = Sequential([
    LSTM(128, input_shape=(SEQ_LEN, len(chars))),
    Dense(len(chars), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

def sample(preds, temperature=0.8):
    preds = np.log(np.array(preds, dtype=np.float64) + 1e-10) / temperature
    exp_preds = np.exp(preds - preds.max())
    return np.random.choice(len(preds), p=exp_preds/exp_preds.sum())

def generate(epoch, _):
    if (epoch+1) % 5 != 0: return
    seed = text[random.randint(0, len(text)-SEQ_LEN-1):][:SEQ_LEN]
    generated = seed
    for _ in range(200):
        x_pred = np.zeros((1, SEQ_LEN, len(chars)))
        for t, c in enumerate(seed):
            x_pred[0, t, char2idx[c]] = 1
        idx = sample(model.predict(x_pred, verbose=0)[0])
        next_char = idx2char[idx]
        generated += next_char
        seed = seed[1:] + next_char
    print(f"\n── Epoch {epoch+1} sample ──\n{generated}\n")

model.fit(X, y, batch_size=128, epochs=30,
          callbacks=[LambdaCallback(on_epoch_end=generate)])
