#!/usr/bin/env python
# src/train_with_tuning.py
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources (only the first time)
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    """
    Refined cleaning function:
    - Converts text to lowercase
    - Removes punctuation
    - Tokenizes the text
    - Removes NLTK stopwords
    Returns the processed text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def main():
    # Simple demo: process a sample text
    sample_text = "Breaking News: The stock market crashed today!"
    print("Original:", sample_text)
    processed = clean_text(sample_text)
    print("Processed:", processed)

if __name__ == '__main__':
    main()