import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Define the path to the processed data file
DATA_PATH = os.path.join("data", "processed", "FakeAndRealNews_processed.csv")

def load_data(path):
    """Load processed data from a CSV file."""
    df = pd.read_csv(path)
    return df

def feature_engineering(df, text_column='combined_text', max_features=5000):
    """
    Convert text data to numerical features using TF-IDF.
    :param df: DataFrame containing processed data.
    :param text_column: The column in df with the cleaned text.
    :param max_features: Maximum number of features for the vectorizer.
    :return: The fitted TF-IDF vectorizer and the transformed feature matrix.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df[text_column])
    return vectorizer, X_tfidf

def train_model(X, y, test_size=0.2, random_state=42, max_iter=300):
    """
    Split the data, train a Logistic Regression model and evaluate its performance.
    :param X: Feature matrix.
    :param y: Labels.
    :param test_size: Fraction of data for testing.
    :param random_state: Random state for reproducibility.
    :param max_iter: Maximum iterations for Logistic Regression.
    :return: Trained model, X_test, y_test and predictions.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluation display
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test, y_pred

def save_artifacts(model, vectorizer):
    """
    Save trained model and TF-IDF vectorizer for later use.
    """
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "classical_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Trained model saved to {model_path}")
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")

def main():
    # Step 1: Load processed data
    df = load_data(DATA_PATH)
    
    # Ensure the necessary columns exist
    if 'combined_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Columns 'combined_text' and/or 'label' are missing from the dataset.")

    print("Data loaded successfully. Number of records:", len(df))
    
    # Step 2: Feature Engineering using TF-IDF
    vectorizer, X_tfidf = feature_engineering(df, text_column='combined_text')
    print("TF-IDF features generated. Shape:", X_tfidf.shape)
    
    # Step 3: Prepare labels
    y = df['label']
    
    # Step 4: Train the model and evaluate
    model, X_test, y_test, y_pred = train_model(X_tfidf, y)
    
    # Step 5: Save the model and TF-IDF vectorizer for future use
    save_artifacts(model, vectorizer)

if __name__ == "__main__":
    main()