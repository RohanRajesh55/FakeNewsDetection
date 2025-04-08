#!/usr/bin/env python
# src/train_ensemble.py
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths to production model and vectorizer
PROD_MODEL_PATH = os.path.join("models", "classical_model.pkl")
PROD_VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
# Paths to candidate model and vectorizer
CANDIDATE_MODEL_PATH = os.path.join("models", "candidate_model.pkl")
CANDIDATE_VECTORIZER_PATH = os.path.join("models", "candidate_tfidf_vectorizer.pkl")

# Load production model and vectorizer
prod_model = joblib.load(PROD_MODEL_PATH)
prod_vectorizer = joblib.load(PROD_VECTORIZER_PATH)

candidate_model = None
candidate_vectorizer = None
if os.path.exists(CANDIDATE_MODEL_PATH) and os.path.exists(CANDIDATE_VECTORIZER_PATH):
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)
    candidate_vectorizer = joblib.load(CANDIDATE_VECTORIZER_PATH)

def ensemble_predict(news_text):
    """
    Process the input news text with both production and candidate vectorizers,
    obtain probability predictions from both models, and average them.
    Returns the final predicted label (1 for Real, 0 for Fake) along with the probability.
    """
    # Get production model probability (assume probability for class 1 is index 1)
    X_prod = prod_vectorizer.transform([news_text])
    prod_prob = prod_model.predict_proba(X_prod)[0, 1]
    
    if candidate_model and candidate_vectorizer:
        X_cand = candidate_vectorizer.transform([news_text])
        cand_prob = candidate_model.predict_proba(X_cand)[0, 1]
        final_prob = (prod_prob + cand_prob) / 2.0
    else:
        final_prob = prod_prob
        
    final_label = 1 if final_prob >= 0.5 else 0
    return final_label, final_prob

if __name__ == "__main__":
    sample_text = "Breaking news: The stock market is soaring after major policy changes."
    label, probability = ensemble_predict(sample_text)
    print(f"Ensemble Prediction: {label} with probability {probability:.3f}")