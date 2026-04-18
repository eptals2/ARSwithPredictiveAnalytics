"""Module for job suitability prediction using Flask and machine learning."""

import os
from utilities.pre_processing import preprocess_text
from utilities.text_extraction import extract_text

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets of words."""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

# Main function to compare resumes with job descriptions
def calculate_resume_similarities(job_path, resume_paths):
    """Compute Jaccard similarity scores for resumes against job description."""
    
    job_text = extract_text(job_path) or ""  # Handle None case
    job_tokens = set(preprocess_text(job_text).split())  # Convert to set
    
    results = []

    for resume_path in resume_paths:
        resume_text = extract_text(resume_path) or ""  # Handle None case
        resume_tokens = set(preprocess_text(resume_text).split())  # Convert to set

        similarity_score = jaccard_similarity(resume_tokens, job_tokens)

        # Find common words (intersection of job & resume words)
        intersecting_words = list(resume_tokens.intersection(job_tokens))

        results.append({
            'filename': os.path.basename(resume_path),
            'similarity': round(similarity_score, 2),
            'resume_tokens': list(resume_tokens),
            'intersecting_words': intersecting_words  # Words common in job & resume
        })

    # Sort results by highest similarity score
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        'job_tokens': list(job_tokens),
        'resume_comparisons': results
    }
