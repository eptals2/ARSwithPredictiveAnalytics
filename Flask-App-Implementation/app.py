import pickle
import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from sklearn.metrics.pairwise import cosine_similarity
from utilities.preprocessing import preprocess_text
from utilities.text_extractor import extract_text_from_resume
from utilities.token_generator import generate_secret_key
from utilities.entity_matching import EntityMatcher
import os
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
app.config['SECRET_KEY'] = generate_secret_key()

# Load pre-trained models and vectorizer
try:
    with open("models/XGBoost-trained-model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/XGBoost-trained-model/label_encoder.pkl", "rb") as f:
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

# Initialize EntityMatcher with the model path
MODEL_PATH = "models/RoBERTa-fine-tuned-model"
try:
    entity_matcher = EntityMatcher(MODEL_PATH)
    logging.info("EntityMatcher initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing EntityMatcher: {e}")
    raise e

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to assess c
@app.route('/')
def index():
    return render_template('login_page.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    try:
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if username == 'admin' and password == 'admin':
                session['username'] = username
                return redirect(url_for('home'))
            return render_template('login_page.html', error='Invalid username or password')
    except Exception as e:
        return str(e)

@app.route('/home')
def home():
    return render_template('index.html')


@app.route("/upload", methods=["POST"])
def upload_file():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    resume_file = request.files["resume"]  # This is a `FileStorage` object
    extracted_text = extract_text(resume_file)  # Use imported function

    return jsonify({"text": extracted_text})


@app.route("/rank-resumes", methods=["POST"])
def rank_resumes():
    try:
        logging.info("Received request for resume ranking with combined scoring.")

        if "job_requirement" not in request.files or "resumes" not in request.files:
            logging.warning("Missing job description or resumes in request.")
            return jsonify({"success": False, "message": "Missing files"}), 400

        job_file = request.files["job_requirement"]
        resume_files = request.files.getlist("resumes")

        if job_file.filename == "" or not resume_files:
            logging.warning("Empty job description or no resumes provided.")
            return jsonify({"success": False, "message": "Missing files"}), 400

        # Extract text from files
        job_text = extract_text_from_resume(job_file)
        resume_texts = [extract_text_from_resume(resume) for resume in resume_files]

        # Initialize EntityMatcher with model paths
        MODEL_PATH = "models/RoBERTa-fine-tuned-model"
        XGBOOST_PATH = "models/XGBoost-trained-model/xgboost_model.pkl"
        matcher = EntityMatcher(MODEL_PATH, xgboost_model_path=XGBOOST_PATH)

        # Process each resume
        rankings = []
        for i, resume_text in enumerate(resume_texts):
            # Get RoBERTa entity analysis and role prediction
            results = matcher.analyze_resume(resume_text, job_text)
            
            # Add to rankings
            rankings.append({
                'resume_filename': resume_files[i].filename,
                'overall_score': results['overall_match'],
                'xgboost_score': results.get('role_confidence', 0.0),  # Default to 0 if not present
                'jaccard_score': results.get('entity_analysis', {}).get('SKILLS', {}).get('matching_score', 0.0),  # Default to 0 if not present
                'score_breakdown': results.get('score_breakdown', {
                    'skills': 0.0,
                    'experience': 0.0,
                    'education': 0.0,
                    'certification': 0.0
                }),
                'suitability_status': results['suitability_status'],
                'entity_analysis': results.get('entity_analysis', {})
            })

        # Sort by overall score
        rankings.sort(key=lambda x: x['overall_score'], reverse=True)

        response = {
            "success": True,
            "rankings": rankings
        }

        logging.info("Ranking completed successfully.")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in ranking resumes: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/match', methods=['POST'])
def match():
    try:
        resume_file = request.files.get('resume')
        job_file = request.files.get('job_requirements')

        if not resume_file or not job_file:
            return jsonify({"success": False, "message": "Both resume and job requirements files are required"}), 400

        # Extract text from files
        resume_text = extract_text(resume_file)
        job_text = extract_text(job_file)

        # Initialize EntityMatcher with model path
        matcher = EntityMatcher('models/RoBERTa-fine-tuned-model')
        
        # Get RoBERTa scores and entity analysis
        results = matcher.analyze_resume(resume_text, job_text)
        
        # Get Jaccard similarity scores
        jaccard_score = matcher.calculate_jaccard_similarity(resume_text, job_text)
        
        # Get XGBoost prediction score
        xgboost_score = matcher.predict_with_xgboost(resume_text)
        
        # Calculate overall score
        overall_score = matcher.calculate_overall_score(
            roberta_score=results['roberta_score'],
            jaccard_score=jaccard_score,
            xgboost_score=xgboost_score,
            entity_analysis=results['entity_analysis']
        )

        return jsonify({
            "success": True,
            "roberta_score": round(results['roberta_score'] * 100, 1),
            "jaccard_score": round(jaccard_score * 100, 1),
            "xgboost_score": round(xgboost_score * 100, 1),
            "overall_match": overall_score,
            "entity_analysis": results['entity_analysis']
        })

    except Exception as e:
        logging.error(f"Error in entity matching: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        resume_file = request.files.get('resume')
        job_file = request.files.get('job_requirements')

        if not resume_file or not job_file:
            return jsonify({"success": False, "message": "Both resume and job requirements files are required"}), 400

        # Extract text from files
        resume_text = extract_text(resume_file)
        job_text = extract_text(job_file)

        # Analyze using EntityMatcher
        matcher = EntityMatcher('models/RoBERTa-fine-tuned-model')
        
        # Get RoBERTa scores and entity analysis
        results = matcher.analyze_resume(resume_text, job_text)
        
        # Get Jaccard similarity scores
        jaccard_score = matcher.calculate_jaccard_similarity(resume_text, job_text)
        
        # Get XGBoost prediction score
        xgboost_score = matcher.predict_with_xgboost(resume_text)
        
        # Calculate overall score
        overall_score = matcher.calculate_overall_score(
            roberta_score=results['roberta_score'],
            jaccard_score=jaccard_score,
            xgboost_score=xgboost_score,
            entity_analysis=results['entity_analysis']
        )

        return jsonify({
            'status': 'success',
            'roberta_score': round(results['roberta_score'] * 100, 1),
            'jaccard_score': round(jaccard_score * 100, 1),
            'xgboost_score': round(xgboost_score * 100, 1),
            'overall_match': overall_score,
            'entity_analysis': results['entity_analysis']
        })

    except Exception as e:
        logging.error(f"Error in entity matching: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=10000)
