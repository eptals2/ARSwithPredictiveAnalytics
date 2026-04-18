"""Module for job suitability prediction using Flask and machine learning."""

import os
from utilities.pre_processing import preprocess_text
from utilities.text_extraction import extract_text

def jaccard_similarity(text1, text2):
    """Compute Jaccard similarity between two preprocessed texts."""
    set1 = set(preprocess_text(text1).split())  # Convert preprocessed text into a set of words
    set2 = set(preprocess_text(text2).split())

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union != 0 else 0.0
