print("🚀 Script started...")

import os
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK stopwords
nltk.download('stopwords')

# Parameters
MAX_VOCAB = 5000
MAX_LEN = 50
EMBEDDING_DIM = 64
EPOCHS = 5

# Step 1: Load and Preprocess Dataset
print("📥 Loading dataset...")
df = pd.read_csv("data/news_headlines.csv", encoding="ISO-8859-1")
df = df[['Date', 'Label'] + [f'Top{i}' for i in range(1, 26)]]

# Combine all 25 headlines into one string per row
df['Combined'] = df[[f'Top{i}' for i in range(1, 26)]].astype(str).agg(' '.join, axis=1)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

print("🧹 Cleaning text...")
df['Clean'] = df['Combined'].apply(clean_text)

# Convert labels to integers
df['Label'] = df['Label'].astype(int)

# Step 2: Tokenize and Pad
print("🔢 Tokenizing and padding...")
tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(df['Clean'])
X = tokenizer.texts_to_sequences(df['Clean'])
X = pad_sequences(X, maxlen=MAX_LEN)
y = df['Label'].values

# Step 3: Train/Test Split
print("📊 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build LSTM Model
print("🧠 Building model...")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_VOCAB, EMBEDDING_DIM, input_length=MAX_LEN),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("✅ Model compiled")

# Step 5: Train
print("🚀 Training model...")
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=32, callbacks=[early_stop])

# Step 6: Evaluate
print("📈 Evaluating model...")
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# Step 7: Save Accuracy Plot
print("📊 Saving accuracy plot...")
os.makedirs("reports", exist_ok=True)

plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("LSTM Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("reports/news_sentiment_lstm.png")
plt.show()
# Step 8: Save model and tokenizer
print("💾 Saving model and tokenizer...")
os.makedirs("models", exist_ok=True)
model.save("models/lstm_sentiment_model.h5")

import pickle
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model and tokenizer saved.")
