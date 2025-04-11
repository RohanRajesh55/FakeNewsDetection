import os
import shutil
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("data", "processed", "FakeAndRealNews_processed.csv")
PROD_MODEL_PATH = os.path.join("models", "classical_model.pkl")
PROD_VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
CANDIDATE_MODEL_PATH = os.path.join("models", "candidate_model.pkl")
CANDIDATE_VECTORIZER_PATH = os.path.join("models", "candidate_tfidf_vectorizer.pkl")

IMPROVEMENT_MARGIN = 0.01

def load_data():
    
    df = pd.read_csv(DATA_PATH)
    if 'combined_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Expected columns 'combined_text' and 'label' not found.")
    return df

def evaluate_model(model, vectorizer, X_val_text, y_val):
   
    X_val = vectorizer.transform(X_val_text)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='weighted')

def promote_candidate():
   
    if not (os.path.exists(CANDIDATE_MODEL_PATH) and os.path.exists(CANDIDATE_VECTORIZER_PATH)):
        print("Candidate model files not found. Promotion aborted.")
        return

    df = load_data()

    X_train, X_val, y_train, y_val = train_test_split(df['combined_text'], df['label'], 
                                                      test_size=0.2, random_state=42, stratify=df['label'])
    prod_model = joblib.load(PROD_MODEL_PATH)
    prod_vectorizer = joblib.load(PROD_VECTORIZER_PATH)
    prod_f1 = evaluate_model(prod_model, prod_vectorizer, X_val, y_val)
    print(f"Production Model F1 Score: {prod_f1:.3f}")


    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)
    candidate_vectorizer = joblib.load(CANDIDATE_VECTORIZER_PATH)
    candidate_f1 = evaluate_model(candidate_model, candidate_vectorizer, X_val, y_val)
    print(f"Candidate Model F1 Score: {candidate_f1:.3f}")


    if candidate_f1 >= prod_f1 + IMPROVEMENT_MARGIN:
        print("Candidate model outperforms production. Promoting candidate model...")

        shutil.copy(CANDIDATE_MODEL_PATH, PROD_MODEL_PATH)
        shutil.copy(CANDIDATE_VECTORIZER_PATH, PROD_VECTORIZER_PATH)
        print("Promotion successful: Production model has been updated.")
    else:
        print("Candidate model not significantly better. No promotion performed.")

if __name__ == "__main__":
    promote_candidate()