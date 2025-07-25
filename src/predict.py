import joblib

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_news(text):
    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)
    return prediction[0]
