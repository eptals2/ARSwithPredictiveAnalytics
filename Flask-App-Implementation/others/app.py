import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utilities.text_extraction import extract_text  # Ensure you have this utility
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load trained model and encoders
with open("Flask-App-Implementation/models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("Flask-App-Implementation/models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("Flask-App-Implementation/models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Allowed file types
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "files[]" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    files = request.files.getlist("files[]")
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)

            # Extract resume text
            resume_text = extract_text(file_path)

            # Transform text into TF-IDF features
            resume_tfidf = vectorizer.transform([resume_text])

            # Predict job role
            probabilities = model.predict_proba(resume_tfidf)[0]
            top_indices = np.argsort(probabilities)[::-1][:3]  # Get top 3 job roles
            top_roles = label_encoder.inverse_transform(top_indices)
            top_scores = probabilities[top_indices] * 100  # Convert to percentage

            # Generate skills and experience analysis
            analysis = analyze_resume(resume_text)

            # Store result
            results.append({
                "filename": filename,
                "top_jobs": [{ "role": top_roles[i], "score": f"{top_scores[i]:.2f}%" } for i in range(len(top_roles))],
                "analysis": analysis
            })

    return jsonify({"resumes": results})

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
