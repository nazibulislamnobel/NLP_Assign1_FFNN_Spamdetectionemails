!pip install nltk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Softmax

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

#setup
nltk.download(['stopwords', 'wordnet'])
#tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # Faster lookups
#preview
print("Stopwords sample:", list(stop_words)[:5])

raw_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/emails.csv')
working_df = raw_data.drop(["Email No.", "Prediction"], axis=1)

#data check
working_df.sample(15)

print(type(stop_words))

non_stopword_cols = [col for col in working_df.columns if col not in stop_words]

#diagnostics
print("Filtered columns:", non_stopword_cols)
print("Original columns:", len(working_df.columns))
print("Filtered count:", len(non_stopword_cols))

dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/emails.csv')
df = pd.DataFrame(dataset, columns=working_df.columns)  # Using previously filtered columns

print("Final feature count:", len(working_df.columns))

Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]

main_dataset_train, main_dataset_test, main_predication_train, main_predication_test = train_test_split(
    Main_dataset, Main_predication_dataset, test_size=0.05, random_state=101
)

scaler = StandardScaler()
Main_train = scaler.fit_transform(main_dataset_train)
Main_test = scaler.transform(main_dataset_test)

#neural network architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(Main_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

#model training
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#train model
history = model.fit(
    Main_train, main_predication_train,
    epochs=20,
    batch_size=32,
    validation_data=(Main_test, main_predication_test),
    verbose=1
)

#predictions
main_trained = (model.predict(Main_test) > 0.5).astype(int)

#model performance
conf_matrix = confusion_matrix(main_predication_test, main_trained)
f1 = f1_score(main_predication_test, main_trained)
class_report = classification_report(main_predication_test, main_trained)

#evaluation metrics
print("\nModel Evaluation Results")
print("="*50)
print("Confusion Matrix:\n", conf_matrix)
print("\nF1 Score: {:.4f}".format(f1))
print("\nClassification Report:\n", class_report)

plt.figure(figsize=(12, 6))

#accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(alpha=0.3)
plt.legend()

#loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'b--', label='Train Loss')
plt.plot(history.history['val_loss'], 'r--', label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix,
                           classification_report,
                           f1_score)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding,
                                   LSTM, Bidirectional,
                                   Dropout, Flatten)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

max_words = 10000  
max_len = 200

Main_dataset = df.copy()
Main_predication_dataset = dataset["Prediction"]

X_train, X_test, y_train, y_test = train_test_split(
    Main_dataset,
    Main_predication_dataset,
    test_size=0.2,
    random_state=56
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nConfusion Matrix:\n{conf_matrix}")
print(f"\nF1 Score: {f1:.4f}")
print(f"\nClassification Report:\n{class_report}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-', linewidth=2, label='Train Accuracy')
plt.plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
plt.title('Model Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'b--', linewidth=2, label='Train Loss')
plt.plot(history.history['val_loss'], 'r--', linewidth=2, label='Validation Loss')
plt.title('Model Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()