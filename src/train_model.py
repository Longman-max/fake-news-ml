import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Define base paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_path, "data", "fake_or_real_news.csv")
models_path = os.path.join(base_path, "models")

# Load data
df = pd.read_csv(data_path)
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs(models_path, exist_ok=True)
joblib.dump(model, os.path.join(models_path, "model.pkl"))
joblib.dump(vectorizer, os.path.join(models_path, "vectorizer.pkl"))

print("✅ Model and vectorizer saved to 'models/' folder.")
