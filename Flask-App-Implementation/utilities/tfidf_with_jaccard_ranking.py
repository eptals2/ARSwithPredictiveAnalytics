import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utilities.pre_processing import preprocess_text
from utilities.jaccard_similarity_scoring import jaccard_similarity

# Load pre-trained vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def rank_resumes(job_text, resume_texts):
    """Ranks resumes based on TF-IDF similarity and Jaccard similarity."""

    # Preprocess job description
    job_cleaned = preprocess_text(job_text)
    
    # Preprocess each resume
    resume_cleaned_texts = [preprocess_text(resume) for resume in resume_texts]

    # Compute TF-IDF vectors
    all_texts = [job_cleaned] + resume_cleaned_texts
    tfidf_matrix = vectorizer.transform(all_texts)
    
    # Calculate TF-IDF cosine similarity
    job_vector = tfidf_matrix[0]  # Job requirement vector
    resume_vectors = tfidf_matrix[1:]  # Resume vectors
    tfidf_similarities = cosine_similarity(job_vector, resume_vectors).flatten()

    # Calculate Jaccard similarity
    jaccard_similarities = [
        jaccard_similarity(job_cleaned, resume) for resume in resume_cleaned_texts
    ]

    # Combine TF-IDF and Jaccard scores (weighted sum)
    final_scores = [
        (0.6 * tfidf_sim) + (0.4 * jaccard_sim) 
        for tfidf_sim, jaccard_sim in zip(tfidf_similarities, jaccard_similarities)
    ]

    # Rank resumes based on final scores
    ranked_resumes = sorted(
        enumerate(final_scores), key=lambda x: x[1], reverse=True
    )

    # Prepare results
    results = [
        {"resume_index": idx, "score": round(score * 100, 2)}  # Convert to percentage
        for idx, score in ranked_resumes
    ]

    return results
