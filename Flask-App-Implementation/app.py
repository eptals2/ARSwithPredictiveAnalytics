import pickle
import logging
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from utilities.pre_processing import preprocess_text
from utilities.jaccard_similarity_scoring import jaccard_similarity
from utilities.text_extraction import extract_text

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Load pre-trained models and vectorizer
try:
    with open("C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Flask-App-Implementation/models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Flask-App-Implementation/models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise e

# Demo model class for simulation
class DemoXGBoostModel:
    def predict(self, X):
        # Simulate predictions for demonstration
        import numpy as np
        possible_predictions = [0, 1, 2]  # Assuming we have 3 possible job categories
        return np.random.choice(possible_predictions, size=X.shape[0])

xgb_model = DemoXGBoostModel()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    resume_file = request.files["resume"]  # This is a `FileStorage` object
    extracted_text = extract_text(resume_file)  # Use imported function

    return jsonify({"text": extracted_text})


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

        logging.debug(f"Extracted job text: {job_text[:500]}")
        logging.debug(f"Extracted {len(resume_texts)} resumes.")

        # Preprocess text
        job_cleaned = preprocess_text(job_text)
        resume_cleaned_texts = [preprocess_text(resume) for resume in resume_texts]

        # Fit the vectorizer with the job and resumes
        all_texts = [job_cleaned] + resume_cleaned_texts
        vectorizer.fit(all_texts)  # Fit the vectorizer
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

@app.route("/analyze-suitability", methods=["POST"])
def analyze_suitability():
    try:
        logging.info("Received request for suitability analysis.")
        data = request.json

        if "filtered_resumes" not in data or "job_requirement" not in data:
            logging.warning("Missing data in request.")
            return jsonify({"success": False, "message": "Missing data"}), 400

        job_text = data["job_requirement"]
        resume_texts = data["filtered_resumes"]

        all_texts = [job_text] + resume_texts
        tfidf_matrix = vectorizer.transform(all_texts)
        resume_vectors = tfidf_matrix[1:]

        if resume_vectors.shape[0] == 0:
            logging.warning("No valid resumes to analyze.")
            return jsonify({"success": False, "message": "No valid resumes to analyze"}), 400

        predictions = xgb_model.predict(resume_vectors)
        predicted_labels = label_encoder.inverse_transform(predictions)

        alternative_jobs = {job: round((predicted_labels.tolist().count(job) / len(predicted_labels)) * 100, 2) for job in set(predicted_labels)}

        logging.info("Suitability analysis completed successfully.")
        return jsonify({"success": True, "alternative_jobs": alternative_jobs})

    except Exception as e:
        logging.error(f"Error in suitability analysis: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
