"""Module for job suitability prediction using Flask and machine learning."""

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import inflect
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download required NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Initialize lemmatizer & inflect engine for number conversion
lemmatizer = WordNetLemmatizer()
inflect_engine = inflect.engine()

def convert_numbers_to_words(text):
    """Convert numeric digits in text to words (e.g., '3' → 'three')."""
    words = text.split()
    converted_words = []
    
    for word in words:
        if word.isdigit():  # Check if it's a number
            word = inflect_engine.number_to_words(int(word))  # Convert number to words
        converted_words.append(word)
    
    return " ".join(converted_words)

def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, lemmatize, and normalize numbers."""
    text = text.lower()  # Convert to lowercase
    text = convert_numbers_to_words(text)  # Convert numbers to words
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize into words

    # Lemmatization & stopword removal
    processed_words = [
        lemmatizer.lemmatize(word) for word in words if word not in ENGLISH_STOP_WORDS
    ]
    
    return " ".join(processed_words)  # Return as a string, not a set
