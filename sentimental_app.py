# app.py

import streamlit as st
import pandas as pd
import joblib

# -------------------------------
#  Load model pipeline (cached)
# -------------------------------
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("sentiment_char_svm_pipeline.joblib")
    except FileNotFoundError:
        st.error("❌ Could not load the model file.")
        return None

model = load_pipeline()

# -------------------------------
# App layout & UI
# -------------------------------
st.set_page_config(page_title="Sentiment Classifier", page_icon="🧠")
st.title("🧠 Amazon Review Sentiment Classifier")
st.markdown("Use this tool to predict whether a customer review expresses **positive** or **negative** sentiment.")

# -------------------------------
# Text input
# -------------------------------
review = st.text_area("✍️ Enter a product review:", height=150)

# -------------------------------
#  Predict button
# -------------------------------
if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter a review before submitting.")
    elif model is not None:
        try:
            # Convert to DataFrame or Series to match pipeline expectations
            input_series = pd.Series([review])
            prediction = model.predict(input_series)[0]

            # Interpret prediction
            sentiment = "👍 Positive" if prediction == 1 else "👎 Negative"
            st.success(f"**Predicted Sentiment:** {sentiment}")

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")