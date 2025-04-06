#!/usr/bin/env python
import os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources (only needs to be done once)
nltk.download('stopwords')
nltk.download('punkt')

# Define directory paths
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

def load_raw_datasets(fake_filename="Fake.csv", true_filename="True.csv"):
    """
    Load the fake and true news datasets from the raw directory
    and add a label column where 0 = Fake and 1 = Real.
    """
    fake_path = os.path.join(RAW_DATA_DIR, fake_filename)
    true_path = os.path.join(RAW_DATA_DIR, true_filename)
    
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    
    df_fake['label'] = 0  # Assign 0 for fake news data
    df_true['label'] = 1  # Assign 1 for true news data
    
    return df_fake, df_true

def merge_datasets(df_fake, df_true):
    """Merge fake and true datasets into a single DataFrame."""
    return pd.concat([df_fake, df_true], ignore_index=True)

def clean_text(text):
    """
    Clean input text: lowercasing, punctuation removal, tokenization, and stopword removal.
    Returns cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def combine_text_columns(row, columns=["title", "text"]):
    """
    Combine specified text columns of a DataFrame row into a single string.
    Only includes columns that exist and have non-null values.
    """
    return " ".join([str(row[col]) for col in columns if col in row and pd.notnull(row[col])])

def preprocess_dataset(fake_filename="Fake.csv", true_filename="True.csv", 
                         output_filename="FakeAndRealNews_processed.csv",
                         text_columns=["title", "text"], sample_size=None):
    """
    Load raw datasets, merge them, combine text columns as needed,
    clean the text and save the processed DataFrame to a CSV file.
    
    Parameters:
      - fake_filename: Name of the fake news CSV file.
      - true_filename: Name of the real news CSV file.
      - output_filename: The resulting processed CSV file name.
      - text_columns: List of columns to combine for text processing.
      - sample_size: (Optional) Number of random samples to process for development.
    """
    # Load datasets
    df_fake, df_true = load_raw_datasets(fake_filename, true_filename)
    df_merged = merge_datasets(df_fake, df_true)
    
    # Optionally, work on a subsample for faster iteration on low-resource hardware.
    if sample_size is not None:
        df_merged = df_merged.sample(n=sample_size, random_state=42)
    
    # If both text_columns exist, create a new column combining them.
    if all(col in df_merged.columns for col in text_columns):
        df_merged['combined_text'] = df_merged.apply(lambda row: combine_text_columns(row, text_columns), axis=1)
        target_column = 'combined_text'
    elif "text" in df_merged.columns:
        target_column = "text"
    else:
        raise ValueError("No valid text column found for processing. Check your dataset columns.")
    
    # Clean the text of the target column
    df_merged[target_column] = df_merged[target_column].apply(clean_text)
    
    # Remove rows where text is missing or empty after cleaning
    df_merged.dropna(subset=[target_column], inplace=True)
    
    # Ensure that the processed data directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, output_filename)
    
    # Save the processed DataFrame to a CSV file
    df_merged.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Optionally, use sample_size for faster debugging. Remove sample_size parameter for full dataset.
    preprocess_dataset(sample_size=5000)