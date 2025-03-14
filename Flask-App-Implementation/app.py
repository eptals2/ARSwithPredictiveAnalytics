from flask import Flask, render_template, request, jsonify, redirect, session, url_for
import os
from utilities.token_generator import generate_secret_key
from utilities.text_extractor import extract_text_from_resume
from utilities.preprocessing import preprocess_text
from utilities.RoBERTa_NER import extract_entities
# from utilities.XGboost_assessment import assess_candidate


app = Flask(__name__)
app.config['SECRET_KEY'] = generate_secret_key()

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
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_resumes():
    """Handle multiple resume uploads and extract information"""
    if 'resumes' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    files = request.files.getlist('resumes')
    results = []
    
    for file in files:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        text = extract_text_from_resume(file_path)
        if not text:
            results.append({"filename": file.filename, "error": "Failed to extract text"})
            continue
        
        preprocessed_text = ' '.join(preprocess_text(text))
        extracted_info = extract_entities(preprocessed_text)
        results.append({"filename": file.filename, "extracted_info": extracted_info})
    
    return jsonify(results)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
