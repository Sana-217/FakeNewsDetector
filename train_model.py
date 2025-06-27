import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
print("Current Working Directory:", os.getcwd())

# Load Dataset
df = pd.read_csv('dataset/fake_or_real_news.csv')

# Check structure
print(df.head())

# Features & Labels
X = df['text']      # Full news content
y = df['label']     # FAKE or REAL

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Classifier (PassiveAggressive works well for this)
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Save model & vectorizer
with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("Model saved successfully to 'model/fake_news_model.pkl'")
