#!/usr/bin/env python
import os
import pandas as pd
import string
import nltkn
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Download required nltk resources (only once)
nltk.download('stopwords')
nltk.download('punkt')

# Define directories
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"

##########################
# PREPROCESSING FUNCTIONS
##########################

def clean_text(text):
    """
    Clean the input text by:
      - Lowercasing,
      - Removing punctuation,
      - Tokenizing,
      - Removing NLTK stopwords.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def load_raw_datasets(fake_filename="Fake.csv", true_filename="True.csv"):
    """
    Load fake and true news CSV files and add labels (0 for Fake, 1 for True).
    """
    fake_path = os.path.join(RAW_DATA_DIR, fake_filename)
    true_path = os.path.join(RAW_DATA_DIR, true_filename)
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake['label'] = 0
    df_true['label'] = 1
    return df_fake, df_true

def merge_datasets(df_fake, df_true):
    """Merge fake and true datasets and return combined DataFrame."""
    return pd.concat([df_fake, df_true], ignore_index=True)

def combine_text_columns(row, columns=["title", "text"]):
    """Combine the provided text columns into a single string."""
    return " ".join([str(row[col]) for col in columns if col in row and pd.notnull(row[col])])

def preprocess_dataset(fake_filename="Fake.csv", true_filename="True.csv", 
                         output_filename="FakeAndRealNews_processed.csv",
                         text_columns=["title", "text"], sample_size=None):
    """
    Complete preprocessing: load, merge, combine text columns, clean text, and save processed CSV.
    """
    df_fake, df_true = load_raw_datasets(fake_filename, true_filename)
    df = merge_datasets(df_fake, df_true)
    
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42)
        
    if all(col in df.columns for col in text_columns):
        df['combined_text'] = df.apply(lambda row: combine_text_columns(row, text_columns), axis=1)
        target_column = 'combined_text'
    elif "text" in df.columns:
        target_column = "text"
    else:
        raise ValueError("No valid text column found.")
    
    df[target_column] = df[target_column].apply(clean_text)
    df.dropna(subset=[target_column], inplace=True)
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    return df, target_column

##########################
# FEATURE ENGINEERING & MODEL TRAINING
##########################

def feature_engineering(df, text_column, max_features=5000):
    """
    Convert text to numerical features using TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df[text_column])
    return vectorizer, X_tfidf

def hyperparameter_tuning(X, y):
    """
    Use GridSearchCV to tune hyperparameters for Logistic Regression.
    """
    # Define a minimal parameter grid – feel free to expand this grid.
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
    }
    # LogisticRegression with solver 'liblinear' supports L1 penalty.
    lr = LogisticRegression(solver='liblinear', max_iter=300)
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X, y)
    print("Best parameters:", grid.best_params_)
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test, df_test=None):
    """
    Evaluate the model, display classification report, plot confusion matrix,
    and optionally print misclassified samples.
    """
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Optionally show some misclassified examples, if a DataFrame with original text is provided.
    if df_test is not None:
        df_test = df_test.copy()
        df_test['y_test'] = y_test
        df_test['y_pred'] = y_pred
        misclassified = df_test[df_test['y_test'] != df_test['y_pred']]
        print("\nSome misclassified examples:")
        print(misclassified[['combined_text', 'y_test', 'y_pred']].head(5))

##########################
# MAIN PIPELINE
##########################

def main():
    # Preprocess data (for tuning, we might use full data or a subsample)
    df, target_column = preprocess_dataset(sample_size=5000)  # Remove sample_size for full data
    print("Data loaded successfully. Number of records:", len(df))
    
    # Feature Engineering using TF-IDF
    vectorizer, X_tfidf = feature_engineering(df, target_column)
    print("TF-IDF features generated with shape:", X_tfidf.shape)
    
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_tfidf, y, df, test_size=0.2, random_state=42, stratify=y)
    
    # Hyperparameter Tuning using GridSearchCV
    best_model = hyperparameter_tuning(X_train, y_train)
    
    # Evaluate the tuned model on test data
    evaluate_model(best_model, X_test, y_test, df_test)
    
    # Save the final model and vectorizer for deployment
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "classical_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print("Trained model and TF-IDF vectorizer saved.")

if __name__ == "__main__":
    main()