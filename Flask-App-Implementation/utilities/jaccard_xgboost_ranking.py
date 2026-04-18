"""Module for job suitability prediction using Flask and machine learning."""

def calculate_jaccard_similarity(text1, text2):
    set1 = set(text1.split())  # Convert text1 into a set of words
    set2 = set(text2.split())  # Convert text2 into a set of words
    
    intersection = set1.intersection(set2)  # Common words
    union = set1.union(set2)  # Total unique words
    
    jaccard_score = len(intersection) / len(union) if len(union) > 0 else 0
    
    return jaccard_score, intersection  # Return both score and common words
