import streamlit as st
import joblib
import os

# Paths to model and vectorizer
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please train and save the model as 'models/model.pkl'")
        st.stop()
    if not os.path.exists(VECTORIZER_PATH):
        st.error("Vectorizer file not found. Please save the vectorizer as 'models/vectorizer.pkl'")
        st.stop()
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

# UI
st.title("📰 Fake News Detector")

st.markdown(
    """
    <div style='color: gray; font-size: 15px; padding-bottom: 15px;'>
        ⚠️ <strong>Disclaimer:</strong> This model was trained using data from over 3 years ago. 
        While it still performs well, the model will be updated soon to reflect recent patterns in fake news. 
        <br>
        Results may vary depending on the nature and context of the input.
    </div>
    """, unsafe_allow_html=True
)

st.write("Enter a news article or message below to detect whether it's fake or real.")

input_text = st.text_area("✍️ Paste the message or news content here:", height=200)

if st.button("🔍 Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Vectorize and predict
        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]

        if prediction == 0:
            st.success(f"✅ **REAL News** with {round(proba[0]*100, 2)}% confidence.")
        else:
            st.error(f"⚠️ **FAKE News** with {round(proba[1]*100, 2)}% confidence.")
