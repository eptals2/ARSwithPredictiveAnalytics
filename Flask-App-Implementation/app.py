from flask import Flask, render_template, request, jsonify
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
import os
from utilities.jaccard_RoBERTa_ranking import calculate_resume_similarities

app = Flask(__name__)

# Required for CSRF
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
csrf = CSRFProtect(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
# Load trained model and encoders
with open("./models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("./models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("./models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Allowed file types
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score-resume-using-ner', methods=['POST'])
def score_resume():
    try:
        if 'job_description' not in request.files:
            return jsonify({'success': False, 'message': 'No job description uploaded'}), 400
        
        job_file = request.files['job_description']
        if not job_file or not allowed_file(job_file.filename):
            return jsonify({'success': False, 'message': 'Invalid job description file'}), 400

        # Save job description
        job_filename = secure_filename(job_file.filename)
        job_path = os.path.join(app.config['UPLOAD_FOLDER'], job_filename)
        job_file.save(job_path)

        # Process resumes
        resume_files = request.files.getlist('resumes[]')
        if not resume_files:
            return jsonify({'success': False, 'message': 'No resumes uploaded'}), 400

        resume_paths = []
        for resume in resume_files:
            if resume and allowed_file(resume.filename):
                filename = secure_filename(resume.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume.save(filepath)
                resume_paths.append(filepath)

        if not resume_paths:
            return jsonify({'success': False, 'message': 'No valid resume files uploaded'}), 400

        # Calculate similarities using the function
        analysis_results = calculate_resume_similarities(job_path, resume_paths)

        # Clean up uploaded files
        try:
            os.remove(job_path)
        except Exception as e:
            print(f"Error cleaning up files: {str(e)}")

        return jsonify({'success': True, 'data': analysis_results})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
