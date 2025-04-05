import streamlit as st
import pickle
from src.preprocess import clean_text

# Load the model and vectorizer
@st.cache_resource
def load_model():
    with open("models/fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorize = pickle.load(f)
    return model, vectorize

model, vectorize = load_model()

st.title("📰 Fake News Detection App")
st.markdown("Enter a news title and body text to check whether it's **real or fake**.")

title = st.text_input("News Title", "")
text = st.text_area("News Content", "")

if st.button("Predict"):
    content = title + " " + text
    clean_content = clean_text(content)
    vectorized = vectorize.transform([clean_content]).toarray()
    prediction = model.predict(vectorized)[0]

    label = "🟢 Real" if prediction == 1 else "🔴 Fake"
    st.subheader(f"Prediction: {label}")