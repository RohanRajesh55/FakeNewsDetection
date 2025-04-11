import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

DATA_PATH = os.path.join("data", "processed", "FakeAndRealNews_processed.csv")

def load_data(path):
    df = pd.read_csv(path)
    return df

def feature_engineering(df, text_column='combined_text', max_features=5000):

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df[text_column])
    return vectorizer, X_tfidf

def train_model(X, y, test_size=0.2, random_state=42, max_iter=300):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
 
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test, y_pred

def save_artifacts(model, vectorizer):
 
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "classical_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Trained model saved to {model_path}")
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")

def main():

    df = load_data(DATA_PATH)
    

    if 'combined_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Columns 'combined_text' and/or 'label' are missing from the dataset.")

    print("Data loaded successfully. Number of records:", len(df))
    

    vectorizer, X_tfidf = feature_engineering(df, text_column='combined_text')
    print("TF-IDF features generated. Shape:", X_tfidf.shape)
    

    y = df['label']

    model, X_test, y_test, y_pred = train_model(X_tfidf, y)

    save_artifacts(model, vectorizer)

if __name__ == "__main__":
    main()