"""
=============================================================
  Next Word Prediction - LSTM Model Training Script
  This script handles all 5 tasks and saves artifacts to disk
=============================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import json, pickle, os

# --- Task 1: Dataset Creation ---
# Load the dummy dataset (200+ meaningful English sentences stored in TXT)

DATASET_PATH = "dataset.txt"

print("=" * 60)
print("  TASK 1 : Loading Dataset")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

sentences = [s.strip() for s in raw_text.split("\n") if s.strip()]
print(f"  [OK] Loaded {len(sentences)} sentences from '{DATASET_PATH}'")
print(f"  [OK] Sample sentences:")
for s in sentences[:5]:
    print(f"    - {s}")

# --- Task 2: NLP Preprocessing ---
# Steps: Lowercasing, Tokenization, Vocabulary creation,
#        Sequence generation, Padding sequences

print("\n" + "=" * 60)
print("  TASK 2 : NLP Preprocessing")
print("=" * 60)

# Step 1: Lowercasing
sentences_lower = [s.lower() for s in sentences]
print("  [OK] Lowercasing complete")

# Step 2 & 3: Tokenization + Vocabulary creation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_lower)

total_words = len(tokenizer.word_index) + 1  # +1 for padding token
print(f"  [OK] Tokenization complete")
print(f"  [OK] Total Vocabulary Size : {total_words}")

# Step 4: Sequence generation (n-gram sequences)
input_sequences = []
for line in sentences_lower:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

print(f"  [OK] Total sequences generated : {len(input_sequences)}")

# Step 5: Padding sequences
max_sequence_len = max(len(x) for x in input_sequences)
input_sequences_padded = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

print(f"  [OK] Maximum Sequence Length : {max_sequence_len}")
print(f"  [OK] Sample padded sequences :")
for seq in input_sequences_padded[:3]:
    print(f"    {seq}")

# Split into features (X) and label (y)
X = input_sequences_padded[:, :-1]
y = input_sequences_padded[:, -1]

import tensorflow as tf
y_cat = tf.keras.utils.to_categorical(y, num_classes=total_words)

# --- Task 3: LSTM Model Development ---
# Architecture: Embedding -> LSTM -> Dense (softmax)

print("\n" + "=" * 60)
print("  TASK 3 : Building LSTM Model")
print("=" * 60)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

EPOCHS = 100
print(f"\n  Training for {EPOCHS} epochs ...")
history = model.fit(X, y_cat, epochs=EPOCHS, verbose=1)

# --- Save everything to disk ---

print("\n" + "=" * 60)
print("  Saving Model & Artifacts")
print("=" * 60)

os.makedirs("saved_model", exist_ok=True)

# Save Keras model
model.save("saved_model/next_word_lstm.keras")
print("  [OK] Model saved -> saved_model/next_word_lstm.keras")

# Save tokenizer
with open("saved_model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("  [OK] Tokenizer saved -> saved_model/tokenizer.pkl")

# Save training history
with open("saved_model/history.json", "w") as f:
    json.dump(history.history, f)
print("  [OK] Training history saved -> saved_model/history.json")

# Save metadata
meta = {
    "total_words": total_words,
    "max_sequence_len": max_sequence_len,
    "epochs": EPOCHS,
    "num_sentences": len(sentences),
    "num_sequences": len(input_sequences),
}
with open("saved_model/metadata.json", "w") as f:
    json.dump(meta, f)
print("  [OK] Metadata saved -> saved_model/metadata.json")

# --- Task 4 demo (console) ---
print("\n" + "=" * 60)
print("  TASK 4 : Quick Prediction Test")
print("=" * 60)

test_phrase = "artificial intelligence is"
token_list = tokenizer.texts_to_sequences([test_phrase])[0]
token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
predicted = model.predict(token_list, verbose=0)[0]
predicted_word = ""
for word, index in tokenizer.word_index.items():
    if index == np.argmax(predicted):
        predicted_word = word
        break
print(f'  Input  : "{test_phrase}"')
print(f'  Output : "{predicted_word}"')

# --- Task 5 demo (save plots) ---
print("\n" + "=" * 60)
print("  TASK 5 : Saving Performance Plots")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history["accuracy"], color="#4f8cff", linewidth=2)
ax1.set_title("Training Accuracy", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(history.history["loss"], color="#ff6b6b", linewidth=2)
ax2.set_title("Training Loss", fontsize=14, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("saved_model/training_plots.png", dpi=150)
print("  [OK] Plot saved -> saved_model/training_plots.png")

print("\n  All tasks complete. Run 'streamlit run app.py' to use the UI.\n")
