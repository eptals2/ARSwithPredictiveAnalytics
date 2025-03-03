import pickle
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from utilities.pre_processing import preprocess_text
from utilities.jaccard_similarity_scoring import jaccard_similarity
from utilities.text_extraction import extract_text
import os
import numpy as np
# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tfidf-with-jaccard-ranking", methods=["POST"])
def rank_resumes():
    try:
        logging.info("Received request for TF-IDF + Jaccard ranking.")

        if "job_requirement" not in request.files or "resumes" not in request.files:
            logging.warning("Missing job description or resumes in request.")
            return jsonify({"success": False, "message": "Missing files"}), 400

        job_file = request.files["job_requirement"]
        resume_files = request.files.getlist("resumes")

        if job_file.filename == "":
            logging.warning("Job description file is empty.")
            return jsonify({"success": False, "message": "Job description file is empty"}), 400

        if not resume_files:
            logging.warning("No resumes uploaded.")
            return jsonify({"success": False, "message": "No resumes provided"}), 400

        # Extract text
        job_text = extract_text(job_file)
        resume_texts = [extract_text(resume) for resume in resume_files]

        # Preprocess text
        job_cleaned = preprocess_text(job_text)
        resume_cleaned_texts = [preprocess_text(resume) for resume in resume_texts]

        # logging.debug(f"Extracted {len(resume_texts)} resumes.")
        # logging.debug(f"Extracted resume text: {resume_texts[:5000]}")
        # logging.debug(f"Extracted job text: {job_text[:5000]}")
        
        # Convert to TF-IDF vectors
        all_texts = [job_cleaned] + resume_cleaned_texts
        tfidf_matrix = vectorizer.transform(all_texts)
        job_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]

        if resume_vectors.shape[0] == 0:
            logging.warning("No valid resumes to rank.")
            return jsonify({"success": False, "message": "No valid resumes to rank"}), 400

        # Compute similarities
        tfidf_similarities = cosine_similarity(job_vector.reshape(1, -1), resume_vectors).flatten()
        jaccard_similarities = [jaccard_similarity(job_cleaned, resume) for resume in resume_cleaned_texts]

        # Combine scores
        final_scores = [(0.8 * tfidf_sim) + (0.2 * jaccard_sim) for tfidf_sim, jaccard_sim in zip(tfidf_similarities, jaccard_similarities)]

        # Separate failed resumes
        ranked_resumes = []
        for idx, score in enumerate(final_scores):
            filename = resume_files[idx].filename
            if score < 0.50:  # Move failed resumes to the failed_resumes folder
                failed_path = os.path.join(FAILED_FOLDER, filename)
                resume_files[idx].save(failed_path)
                logging.info(f"Moved {filename} to failed resumes folder.")
            else:
                ranked_resumes.append((idx, score))

        # Rank resumes
        ranked_resumes = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)

        response = {
            "success": True,
            "rankings": [
                {"resume_filename": resume_files[idx].filename, "score": round(score * 100, 2)}
                for idx, score in ranked_resumes
            ],
        }

        logging.info("Ranking completed successfully.")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in ranking resumes: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

#--------------------------------------
# XGBoost Prediction
#--------------------------------------
# Load pre-trained models and vectorizer
try:
    with open("Flask-App-Implementation/models/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("Flask-App-Implementation/models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("Flask-App-Implementation/models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise e

# Allowed file types
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

# Failed resumes folder
FAILED_FOLDER = "Flask-App-Implementation/failed_resumes"
if not os.path.exists(FAILED_FOLDER):
    os.makedirs(FAILED_FOLDER)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/analyze-failed-resume", methods=["POST"])
def analyze_failed_resume():
    if not os.path.exists(FAILED_FOLDER):
        return jsonify({"success": False, "message": "No failed resumes found."}), 400

    files = [f for f in os.listdir(FAILED_FOLDER) if allowed_file(f)]
    results = []

    if not files:
        return jsonify({"success": False, "message": "No valid resume files in failed folder."}), 400

    for file in files:
        file_path = os.path.join(FAILED_FOLDER, file)

        # Extract resume text
        resume_text = extract_text(file_path)
        if not resume_text:
            print(f"❌ Failed to extract text from {file}")
            continue  # Skip this file

        # Transform text into TF-IDF features
        try:
            resume_tfidf = vectorizer.transform([resume_text])
        except Exception as e:
            print(f"❌ TF-IDF transformation failed for {file}: {e}")
            continue

        # Predict job role using xgb_model
        try:
            probabilities = xgb_model.predict_proba(resume_tfidf)[0]
            top_indices = np.argsort(probabilities)[::-1][:3]  # Get top 3 job roles
            top_roles = label_encoder.inverse_transform(top_indices)
            top_scores = probabilities[top_indices] * 100  # Convert to percentage
        except Exception as e:
            print(f"❌ Prediction failed for {file}: {e}")
            continue

        # Generate skills and experience analysis
        analysis = analyze_resume(resume_text) if "analyze_resume" in globals() else {}

        # Store result
        results.append({
            "filename": file,
            "top_jobs": [{"role": top_roles[i], "score": f"{top_scores[i]:.2f}%"} for i in range(len(top_roles))],
            "analysis": analysis
        })

    return jsonify({"success": True, "data": results}) if results else jsonify({"success": False, "message": "No resumes successfully analyzed."}), 400





def analyze_resume(text):
    """Basic resume analysis - Extracts key skills & experience."""
    skills = ["Python", "Java", "C++", "Machine Learning", "Data Analysis", "Sales", "Excel", "Project Management"]
    experience_keywords = ["years", "months", "developer", "manager", "engineer", "assistant"]
    
    detected_skills = [skill for skill in skills if skill.lower() in text.lower()]
    experience = [word for word in text.split() if word.lower() in experience_keywords]

    return {
        "skills": detected_skills or "Not detected",
        "experience": " ".join(experience) or "Not detected"
    }



if __name__ == "__main__":
    app.run(debug=True)
