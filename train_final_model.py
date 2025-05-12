# model.py
import joblib
import pandas as pd

# Load pipeline (once)
def load_pipeline(path="sentiment_char_svm_pipeline.joblib"):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {path}")

# Predict sentiment
def predict_sentiment(text, model):
    if not text.strip():
        return "⚠️ Invalid input"

    input_series = pd.Series([text])
    prediction = model.predict(input_series)[0]
    return "👍 Positive" if prediction == 1 else "👎 Negative"