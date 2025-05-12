"""
This module contains preprocessing, vectorization, modeling, and evaluation functions
used for sentiment classification on text data.

Note:
Each vectorizer (char, word, n-gram) is paired with a different preprocessing strategy.
Refer to the  Notebook for full explanation.
"""

# -----------------------------------
# IMPORTS
# -----------------------------------

import pandas as pd
import re
import spacy
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# -----------------------------------
# GLOBAL SEED SETUP FOR REPRODUCIBILITY
# -----------------------------------

SEED = 42
np.random.seed(SEED)     # Set seed for numpy operations
random.seed(SEED)        # Set seed for Python's random module

# Load SpaCy language model
nlp = spacy.load("en_core_web_sm")

# -----------------------------------
# TEXT CLEANING FUNCTIONS
# -----------------------------------

def basic_clean(text):
    """
    Apply basic preprocessing: lowercase, remove punctuation/numbers, and strip extra spaces.
    """
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    return re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

def clean_with_spacy(text):
    """
    Use SpaCy to clean text: remove stop words, punctuation, digits, and lemmatize.
    Preserve named entities as-is.
    """
    sent = nlp(text)
    ents = {x.text for x in sent.ents}  # Store named entities
    tokens = []
    for w in sent:
        if w.is_stop or w.is_punct or w.is_digit:
            continue  # Skip stopwords, punctuation, digits
        if w.text in ents:
            tokens.append(w.text)  # Keep entity names unchanged
        else:
            tokens.append(w.lemma_.lower())  # Use lemmatized form
    return ' '.join(tokens)

# -----------------------------------
# DATA PREPROCESSING
# -----------------------------------

def preprocess_text_columns(df):
    """
    Create cleaned text columns using both basic regex and SpaCy.
    """
    print("Applying preprocessing...")
    df['char_text'] = df['reviews.text'].astype(str).apply(basic_clean)     # For char-level TF-IDF
    df['ngram_text'] = df['char_text']                                      # Same as char_text for word n-grams
    df['word_text'] = df['reviews.text'].astype(str).apply(clean_with_spacy)  # Word-level with lemmatization
    return df

# -----------------------------------
# VECTORIZER DEFINITIONS
# -----------------------------------

def get_vectorizers():
    """
    Return a dictionary of TF-IDF vectorizers for character-level, word n-gram, and word-level.
    """
    return {
        "char": TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000),
        "ngram": TfidfVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r'\w{1,}', max_features=5000, stop_words='english'),
        "word": TfidfVectorizer(analyzer='word', ngram_range=(1, 1), token_pattern=r'\w{1,}', max_features=5000, stop_words='english')
    }

# -----------------------------------
# MODEL DEFINITIONS
# -----------------------------------

def get_models():
    """
    Return a dictionary of classification models with balanced class weights.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED),
        "Naive Bayes": MultinomialNB(),  # Naive Bayes doesn't use random_state
        "SVM": LinearSVC(max_iter=2000, class_weight='balanced')  # LinearSVC doesn't support random_state
    }

# -----------------------------------
# MODEL EVALUATION FUNCTION
# -----------------------------------

def evaluate_models(X_train, X_test, y_train, y_test, models, balanced=False):
    """
    Train and evaluate each model. Return metrics like Accuracy, Recall, and F1 scores.
    """
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train the model
        preds = model.predict(X_test)  # Predict on test set
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)  # Get detailed report
        acc = accuracy_score(y_test, preds)  # Get overall accuracy

        results.append({
            "Model": name,
            "Balanced": balanced,  # Whether data was SMOTE-balanced
            "Accuracy": acc,
            "Class 0 Recall": report['0']['recall'],  # Important for minority class
            "Class 0 F1": report['0']['f1-score'],
            "Macro F1": report['macro avg']['f1-score']
        })
    return results

# -----------------------------------
# FULL PIPELINE
# -----------------------------------

def run_pipeline(df):
    """
    Main execution pipeline: preprocess text, vectorize, train models, apply SMOTE, and evaluate.
    """
    df = preprocess_text_columns(df)
    vectorizers = get_vectorizers()
    models = get_models()
    final_results = []

    # Select which version of the cleaned text to use for each vectorizer
    texts = {
        "char": df['char_text'],
        "ngram": df['ngram_text'],
        "word": df['word_text']
    }

    for level, vectorizer in vectorizers.items():
        print(f"\n--- TF-IDF Level: {level} ---")
        X = vectorizer.fit_transform(texts[level])  # Convert text to feature vectors
        y = df['sentiment']  # Binary labels

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )

        # Evaluate on original (imbalanced) data
        original_results = evaluate_models(X_train, X_test, y_train, y_test, models, balanced=False)
        for res in original_results:
            res['Vectorizer'] = level
        final_results.extend(original_results)

        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=SEED)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        # Evaluate again on SMOTE-balanced training data
        balanced_results = evaluate_models(X_train_bal, X_test, y_train_bal, y_test, models, balanced=True)
        for res in balanced_results:
            res['Vectorizer'] = level
        final_results.extend(balanced_results)

    return pd.DataFrame(final_results)

