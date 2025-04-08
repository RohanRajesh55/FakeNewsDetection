#!/usr/bin/env python
# src/automate_model_updates.py
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Set file paths
TRAINING_DATA_PATH = os.path.join("data", "processed", "FakeAndRealNews_processed.csv")
FEEDBACK_FILE = "feedback.log"  # New user feedback is logged here
MODELS_DIR = "models"
CANDIDATE_MODEL_PATH = os.path.join(MODELS_DIR, "candidate_model.pkl")
CANDIDATE_VECTORIZER_PATH = os.path.join(MODELS_DIR, "candidate_tfidf_vectorizer.pkl")

def load_feedback(threshold=2):
    """
    Load aggregated feedback from feedback.log.
    Each line in feedback.log should be tab-delimited: 
        news_text<TAB>prediction<TAB>feedback
    For each news_text, aggregate feedback only if there are at least 'threshold' entries.
    Uses majority vote: if majority says "yes," keep the predicted label;
    if majority says "no," flip the label.
    Returns a DataFrame with columns: 'combined_text' and 'label'.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame()

    feedback_dict = {}
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            news_text, prediction, feedback_val = parts[0].strip(), parts[1].strip(), parts[2].strip().lower()
            if news_text not in feedback_dict:
                feedback_dict[news_text] = {"yes": 0, "no": 0, "prediction": prediction}
            if feedback_val == "yes":
                feedback_dict[news_text]["yes"] += 1
            elif feedback_val == "no":
                feedback_dict[news_text]["no"] += 1

    aggregated_feedback = []
    for news_text, counts in feedback_dict.items():
        total = counts["yes"] + counts["no"]
        if total < threshold:
            continue
        original_pred = counts["prediction"].lower()
        if counts["yes"] >= counts["no"]:
            final_label = 1 if original_pred == "real" else 0
        else:
            final_label = 0 if original_pred == "real" else 1
        aggregated_feedback.append({
            "combined_text": news_text,
            "label": final_label
        })
    return pd.DataFrame(aggregated_feedback)

def load_training_data():
    """Load the original processed training data."""
    return pd.read_csv(TRAINING_DATA_PATH)

def retrain_model():
    """
    Retrain a candidate model using both original training data and aggregated feedback.
    Saves the candidate model and TF-IDF vectorizer.
    """
    df_train = load_training_data()
    df_feedback = load_feedback(threshold=2)
    
    if not df_feedback.empty:
        print("Aggregated feedback found. Combining with main training data...")
        df_combined = pd.concat([df_train, df_feedback], ignore_index=True)
    else:
        df_combined = df_train
        print("No adequate feedback available. Retraining on existing data only.")
    
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Feature Engineering using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df_combined['combined_text'])
    y = df_combined['label']
    
    # Quick validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(solver='liblinear', max_iter=300)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print("Candidate Model F1 Score on Validation set: {:.3f}".format(f1))
    
    # Save candidate model and its vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, CANDIDATE_MODEL_PATH)
    joblib.dump(vectorizer, CANDIDATE_VECTORIZER_PATH)
    print("Candidate model and vectorizer saved successfully.")

if __name__ == "__main__":
    retrain_model()