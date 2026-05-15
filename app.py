import streamlit as st
import joblib

# Load trained model
model = joblib.load("sentiment_model.pkl")

# Title
st.title("IMDB Sentiment Analysis App")

# User input
review = st.text_area("Enter Movie Review")

# Prediction button
if st.button("Predict Sentiment"):

    prediction = model.predict([review])[0]

    if prediction == "positive":
        st.success("Positive Review 😊")

    else:
        st.error("Negative Review 😔")