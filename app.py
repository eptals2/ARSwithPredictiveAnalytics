import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory
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

def generate_secret_key(length=24):
    """Generate a secure random secret key"""
    return secrets.token_hex(length)

def match_entities(resume_entities, job_entities, entity_type):
    """Match entities between resume and job requirements"""
    # Get candidate and requirement entities
    candidate_entities = resume_entities.get(entity_type, [])
    requirement_entities = job_entities.get(entity_type, [])
    
    # Convert to lowercase for case-insensitive matching
    candidate_lower = [entity.lower() for entity in candidate_entities]
    requirement_lower = [entity.lower() for entity in requirement_entities]
    
    # Find matching and missing entities
    matching = []
    for req in requirement_lower:
        if any(req in cand or cand in req for cand in candidate_lower):
            matching.append(req)
    
    missing = [req for req in requirement_lower if req not in matching]
    
    # Calculate Jaccard similarity score
    if requirement_lower and candidate_lower:
        intersection = len(matching)
        union = len(requirement_lower) + len(candidate_lower) - intersection
        similarity = intersection / union if union > 0 else 0.0
    else:
        similarity = 0.0
    
    # Store results
    return {
        'requirements': requirement_entities,
        'candidate': candidate_entities,
        'matching': matching,
        'missing': missing,
        'similarity': similarity
    }

# Models will be loaded by EntityMatcher
logging.info("Starting application...")

# Initialize EntityMatcher with model paths
MODEL_PATH = "models/RoBERTa-fine-tuned-model"
# New XGBoost model path for suitability prediction
XGBOOST_PATH = "models/XGBoost-model/xgboost_model.pkl"
try:
    entity_matcher = EntityMatcher(MODEL_PATH, xgboost_model_path=XGBOOST_PATH)
    logging.info("EntityMatcher initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing EntityMatcher: {e}")
    raise e

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Landing page
@app.route('/')
def index():
    return render_template('login_page.html')
#Log in request
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

    resume_file = request.files["resume"]
    extracted_text = extract_text(resume_file)

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

        if not job_text or not any(resume_texts):
            logging.warning("Could not extract text from one or more files.")
            return jsonify({"success": False, "message": "Could not extract text from uploaded files"}), 400

        # Extract entities from job requirements
        job_entities = entity_matcher.extract_entities(job_text)
        logging.info(f"Extracted job entities: {job_entities}")

        # Process each resume
        results = []
        for i, resume_text in enumerate(resume_texts):
            try:
                # Extract entities from resume
                resume_entities = entity_matcher.extract_entities(resume_text)
                logging.info(f"Extracted resume entities: {resume_entities}")
                
                # Calculate entity matches and scores for ranking
                entity_analysis = {}
                score_breakdown = {}
                
                # Process all entity types for matching
                for entity_type in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
                    entity_analysis[entity_type] = match_entities(resume_entities, job_entities, entity_type)
                    
                    # Add to score breakdown
                    similarity = entity_analysis[entity_type].get('similarity', 0.0)
                    score_breakdown[entity_type.lower()] = similarity
                
                # Process each entity type
                for entity_type in ['SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION', 'AGE', 'GENDER', 'ADDRESS']:
                    # Get candidate and requirement entities
                    candidate_entities = resume_entities.get(entity_type, [])
                    requirement_entities = job_entities.get(entity_type, [])
                    
                    # Convert to lowercase for case-insensitive matching
                    candidate_lower = [entity.lower() for entity in candidate_entities]
                    requirement_lower = [entity.lower() for entity in requirement_entities]
                    
                    # Find matching and missing entities
                    matching = []
                    for req in requirement_lower:
                        if any(req in cand or cand in req for cand in candidate_lower):
                            matching.append(req)
                    
                    missing = [req for req in requirement_lower if req not in matching]
                    
                    # Calculate Jaccard similarity score
                    if requirement_lower and candidate_lower:
                        intersection = len(matching)
                        union = len(requirement_lower) + len(candidate_lower) - intersection
                        similarity = intersection / union if union > 0 else 0.0
                    else:
                        similarity = 0.0
                    
                    # Store results
                    entity_analysis[entity_type] = {
                        'requirements': requirement_entities,
                        'candidate': candidate_entities,
                        'matching': matching,
                        'missing': missing,
                        'matching_score': similarity
                    }
                    
                    # Add to score breakdown
                    if entity_type in ['SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
                        score_breakdown[entity_type.lower()] = similarity
                
                # Calculate overall match score with weighted average
                weights = {
                    'age': 0.10,
                    'gender': 0.10,
                    'address': 0.10,
                    'skills': 0.20,
                    'experience': 0.15,
                    'education': 0.15,
                    'certification': 0.20
                }
                
                overall_score = sum(score_breakdown.get(key, 0) * weight for key, weight in weights.items())
                

                # Add the resume to results with both XGBoost confidence and entity match score
                results.append({
                    'resume_filename': resume_files[i].filename,
                    'overall_score': round(matched_entities_score * 100, 1),  # Use matched entities score for ranking
                    'score_breakdown': {
                        'age': round(score_breakdown.get('age', 0.0) * 100, 1),
                        'gender': round(score_breakdown.get('gender', 0.0) * 100, 1),
                        'address': round(score_breakdown.get('address', 0.0) * 100, 1),
                        'skills': round(score_breakdown.get('skills', 0.0) * 100, 1),
                        'experience': round(score_breakdown.get('experience', 0.0) * 100, 1),
                        'education': round(score_breakdown.get('education', 0.0) * 100, 1),
                        'certification': round(score_breakdown.get('certification', 0.0) * 100, 1)
                    },
                    'suitability_status': suitability_status,  # From XGBoost
                    'entity_analysis': entity_analysis
                })
                
                # Debug log entity analysis
                logging.info(f"Entity analysis for {resume_files[i].filename}: {entity_analysis}")
            except Exception as e:
                logging.error(f"Error processing resume {resume_files[i].filename}: {str(e)}")
                continue

        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Return the ranked results
        return jsonify({
            'success': True,
            'rankings': results
        })

    except Exception as e:
        logging.error(f"Error in ranking resumes: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/match", methods=['POST'])
def match():
    try:
        resume_file = request.files.get('resume')
        job_file = request.files.get('job')
        
        if not resume_file or not job_file:
            return jsonify({
                'success': False,
                'message': 'Missing resume or job requirement file'
            }), 400
            
        # Extract text from files
        resume_text = extract_text(resume_file)
        job_text = extract_text(job_file)
        
        if not resume_text or not job_text:
            return jsonify({
                'success': False,
                'message': 'Could not extract text from one or more files'
            }), 400
        
        # Extract entities from job requirements
        job_entities = entity_matcher.extract_entities(job_text)
        
        # Extract entities from resume
        resume_entities = entity_matcher.extract_entities(resume_text)
        
        # Calculate entity matches and scores
        entity_analysis = {}
        score_breakdown = {}
        
        # Process each entity type
        for entity_type in ['SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION', 'AGE', 'GENDER', 'ADDRESS']:
            # Get candidate and requirement entities
            candidate_entities = resume_entities.get(entity_type, [])
            requirement_entities = job_entities.get(entity_type, [])
            
            # Convert to lowercase for case-insensitive matching
            candidate_lower = [entity.lower() for entity in candidate_entities]
            requirement_lower = [entity.lower() for entity in requirement_entities]
            
            # Find matching and missing entities
            matching = []
            for req in requirement_lower:
                if any(req in cand or cand in req for cand in candidate_lower):
                    matching.append(req)
            
            missing = [req for req in requirement_lower if req not in matching]
            
            # Calculate Jaccard similarity score
            if requirement_lower and candidate_lower:
                intersection = len(matching)
                union = len(requirement_lower) + len(candidate_lower) - intersection
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 0.0
            
            # Store results
            entity_analysis[entity_type] = {
                'requirements': requirement_entities,
                'candidate': candidate_entities,
                'matching': matching,
                'missing': missing,
                'matching_score': similarity
            }
            
            # Add to score breakdown
            if entity_type in ['SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION', 'AGE', 'GENDER', 'ADDRESS']:
                score_breakdown[entity_type.lower()] = similarity
        
        # Calculate overall match score with weighted average
        weights = {
            'age': 0.10,
            'gender': 0.10,
            'address': 0.10,
            'skills': 0.20,
            'experience': 0.15,
            'education': 0.15,
            'certification': 0.20
        }
        
        overall_score = sum(score_breakdown.get(key, 0) * weight for key, weight in weights.items())
        
        # Use XGBoost model for binary suitability prediction
        prediction_result = entity_matcher.predict_job_match(resume_entities)
        suitability_status = prediction_result.get('predicted_job', 'Unknown')
        confidence_score = prediction_result.get('confidence', 0.0)
        
        # Use the suitability prediction directly from the XGBoost model
        if suitability_status == "Not Suitable":
            # If the resume is not suitable, return an empty response
            return jsonify({
                'success': False,
                'message': 'Resume does not meet minimum suitability requirements according to XGBoost model'
            }), 200
        
        # Use the binary suitability prediction directly without additional classification
        
        # Format response
        response = {
            'success': True,
            'overall_score': round(overall_score * 100, 1),
            'score_breakdown': {
                'skills': round(score_breakdown.get('skills', 0.0) * 100, 1),
                'experience': round(score_breakdown.get('experience', 0.0) * 100, 1),
                'education': round(score_breakdown.get('education', 0.0) * 100, 1),
                'certification': round(score_breakdown.get('certification', 0.0) * 100, 1)
            },
            'suitability_status': suitability_status,
            'suitability_prediction': suitability_status,
            'confidence': round(confidence_score * 100, 1),
            'entity_analysis': entity_analysis
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in match endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error analyzing resume: {str(e)}'
        }), 500


@app.route("/analyze", methods=['POST'])
def analyze():
    try:
        resume_file = request.files.get('resume')
        job_file = request.files.get('job')
        
        if not resume_file or not job_file:
            return jsonify({
                'success': False,
                'message': 'Missing resume or job requirement file'
            }), 400
            
        # Extract text from files
        resume_text = extract_text(resume_file)
        job_text = extract_text(job_file)
        
        if not resume_text or not job_text:
            return jsonify({
                'success': False,
                'message': 'Could not extract text from one or more files'
            }), 400
        
        # Extract entities from job requirements
        job_entities = entity_matcher.extract_entities(job_text)
        
        # Extract entities from resume
        resume_entities = entity_matcher.extract_entities(resume_text)
        
        # Use new XGBoost model for suitability prediction
        prediction_result = entity_matcher.predict_job_match(resume_entities)
        suitability_status = prediction_result.get('predicted_job', 'Unknown')
        confidence_score = prediction_result.get('confidence', 0.0)
        
        # Calculate entity matches and scores
        entity_analysis = {}
        score_breakdown = {}
        
        # Process each entity type
        for entity_type in ['SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION', 'AGE', 'GENDER', 'ADDRESS']:
            # Get candidate and requirement entities
            candidate_entities = resume_entities.get(entity_type, [])
            requirement_entities = job_entities.get(entity_type, [])
            
            # Convert to lowercase for case-insensitive matching
            candidate_lower = [entity.lower() for entity in candidate_entities]
            requirement_lower = [entity.lower() for entity in requirement_entities]
            
            # Find matching and missing entities
            matching = []
            for req in requirement_lower:
                if any(req in cand or cand in req for cand in candidate_lower):
                    matching.append(req)
            
            missing = [req for req in requirement_lower if req not in matching]
            
            # Calculate Jaccard similarity score
            if requirement_lower and candidate_lower:
                intersection = len(matching)
                union = len(requirement_lower) + len(candidate_lower) - intersection
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 0.0
            
            # Store results
            entity_analysis[entity_type] = {
                'requirements': requirement_entities,
                'candidate': candidate_entities,
                'matching': matching,
                'missing': missing,
                'matching_score': similarity
            }
            
            # Add to score breakdown
            if entity_type in ['SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION', 'AGE', 'GENDER', 'ADDRESS']:
                score_breakdown[entity_type.lower()] = similarity
        
        # Calculate overall match score with weighted average
        weights = {
            'age': 0.10,
            'gender': 0.10,
            'address': 0.10,
            'skills': 0.20,
            'experience': 0.15,
            'education': 0.15,
            'certification': 0.20
        }
        
        overall_score = sum(score_breakdown.get(key, 0) * weight for key, weight in weights.items())
        
        # Use the suitability prediction directly from the model
        if suitability_status == "Not Suitable":
            # If the resume is not suitable, return an empty response
            return jsonify({
                'success': False,
                'message': 'Resume does not meet minimum suitability requirements'
            }), 200
        
        # Use the binary suitability prediction directly without additional classification
        
        # Format response with suitability prediction and confidence
        response = {
            'success': True,
            'overall_score': round(overall_score * 100, 1),
            'score_breakdown': {
                'skills': round(score_breakdown.get('skills', 0.0) * 100, 1),
                'experience': round(score_breakdown.get('experience', 0.0) * 100, 1),
                'education': round(score_breakdown.get('education', 0.0) * 100, 1),
                'certification': round(score_breakdown.get('certification', 0.0) * 100, 1)
            },
            'suitability_status': suitability_status,
            'suitability_prediction': suitability_status,
            'confidence': round(confidence_score * 100, 1),
            'entity_analysis': entity_analysis
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error analyzing resume: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
