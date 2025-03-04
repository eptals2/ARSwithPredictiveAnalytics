from flask import Flask, render_template, request, jsonify
import os
import re
import PyPDF2
import docx
import torch
from transformers import RobertaTokenizer, RobertaForTokenClassification
import xgboost as xgb
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import json

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load RoBERTa NER model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# This is a placeholder for the actual model
# In a real application, you would fine-tune RoBERTa for NER
# For demonstration purposes, we'll simulate the NER extraction
class SimpleNERModel:
    def __init__(self):
        self.entity_patterns = {
            'AGE': r'\b(?:age[:\s]*(\d+)|\b(\d+)\s*(?:years\s*old|yrs|yr old))\b',
            'GENDER': r'\b(?:gender[:\s]*(male|female|non-binary|other)|(?:male|female|non-binary))\b',
            'ADDRESS': r'\b(?:address[:\s]*([A-Za-z0-9\s,.]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Plaza|Plz|Terrace|Ter|Way)[A-Za-z0-9\s,.]*)|(?:[A-Za-z0-9\s,.]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Plaza|Plz|Terrace|Ter|Way)[A-Za-z0-9\s,.]*))\b',
            'SOFT_SKILLS': r'\b(?:problem-solving|communication|teamwork|leadership|adaptability|creativity|time management|critical thinking|emotional intelligence|work ethic|attention to detail|flexibility|collaboration|interpersonal skills|conflict resolution)\b',
            'HARD_SKILLS': r'\b(?:Python|Java|C\+\+|JavaScript|HTML|CSS|SQL|React|Angular|Vue|Node\.js|Django|Flask|AWS|Azure|GCP|Docker|Kubernetes|Excel|PowerBI|Tableau|R|MATLAB|TensorFlow|PyTorch|Data Analysis|Machine Learning|AI|NLP|Computer Vision|DevOps|CI/CD|Git)\b',
            'EDUCATION_LEVEL': r'\b(?:High School|Associate\'s Degree|Bachelor\'s Degree|Master\'s Degree|PhD|Doctorate|MBA|College Graduate|Undergraduate|Graduate|Postgraduate)\b',
            'COURSE': r'\b(?:Computer Science|Information Technology|Software Engineering|Data Science|Artificial Intelligence|Business Administration|Marketing|Finance|Accounting|Economics|Engineering|Mechanical Engineering|Electrical Engineering|Civil Engineering|Psychology|Biology|Chemistry|Physics|Mathematics|Statistics)\b',
            'EXPERIENCE': r'\b(?:(\d+)\s*(?:years|yrs)(?:\s*of)?\s*experience|experience[:\s]*(\d+)\s*(?:years|yrs))\b',
            'CERTIFICATION': r'\b(?:certification[s]?[:\s]*([\w\s]+)|certified\s+([\w\s]+)|([\w\s]+)\s+certified|([A-Z]+)(?:\s*[-–]\s*|\s+)(?:certification|certified|certificate))\b'
        }
    
    def extract_entities(self, text):
        entities = {
            'age': [],
            'gender': [],
            'address': [],
            'soft_skills': [],
            'hard_skills': [],
            'education_level': [],
            'course': [],
            'experience': [],
            'certification': []
        }
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Extract age
        age_matches = re.finditer(self.entity_patterns['AGE'], text, re.IGNORECASE)
        for match in age_matches:
            if match.group(1):
                entities['age'].append(match.group(1))
            elif match.group(2):
                entities['age'].append(match.group(2))
        
        # Extract gender
        gender_matches = re.finditer(self.entity_patterns['GENDER'], text, re.IGNORECASE)
        for match in gender_matches:
            if match.group(1):
                entities['gender'].append(match.group(1))
            else:
                entities['gender'].append(match.group(0))
        
        # Extract address
        address_matches = re.finditer(self.entity_patterns['ADDRESS'], text, re.IGNORECASE)
        for match in address_matches:
            if match.group(1):
                entities['address'].append(match.group(1).strip())
            elif match.group(0):
                entities['address'].append(match.group(0).strip())
        
        # Extract soft skills
        soft_skills_matches = re.finditer(self.entity_patterns['SOFT_SKILLS'], text, re.IGNORECASE)
        for match in soft_skills_matches:
            entities['soft_skills'].append(match.group(0).lower())
        
        # Extract hard skills
        hard_skills_matches = re.finditer(self.entity_patterns['HARD_SKILLS'], text, re.IGNORECASE)
        for match in hard_skills_matches:
            entities['hard_skills'].append(match.group(0))
        
        # Extract education level
        education_level_matches = re.finditer(self.entity_patterns['EDUCATION_LEVEL'], text, re.IGNORECASE)
        for match in education_level_matches:
            entities['education_level'].append(match.group(0))
        
        # Extract course
        course_matches = re.finditer(self.entity_patterns['COURSE'], text, re.IGNORECASE)
        for match in course_matches:
            entities['course'].append(match.group(0))
        
        # Extract experience
        experience_matches = re.finditer(self.entity_patterns['EXPERIENCE'], text, re.IGNORECASE)
        for match in experience_matches:
            if match.group(1):
                entities['experience'].append(f"{match.group(1)} years")
            elif match.group(2):
                entities['experience'].append(f"{match.group(2)} years")
        
        # Extract certification
        certification_matches = re.finditer(self.entity_patterns['CERTIFICATION'], text, re.IGNORECASE)
        for match in certification_matches:
            for group in match.groups():
                if group:
                    entities['certification'].append(group.strip())
                    break
        
        # Remove duplicates and keep only unique values
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities

# Initialize the NER model
ner_model = SimpleNERModel()

# Simulated XGBoost model for candidate assessment
class CandidateAssessmentModel:
    def __init__(self):
        # In a real application, you would train and load an actual XGBoost model
        # For demonstration, we'll use a simple rule-based approach
        self.skill_weights = {
            'soft_skills': {
                'problem-solving': 0.9,
                'communication': 0.8,
                'teamwork': 0.7,
                'leadership': 0.8,
                'adaptability': 0.7,
                'creativity': 0.6,
                'time management': 0.7,
                'critical thinking': 0.9,
                'emotional intelligence': 0.6,
                'work ethic': 0.8,
                'attention to detail': 0.7,
                'flexibility': 0.6,
                'collaboration': 0.7,
                'interpersonal skills': 0.7,
                'conflict resolution': 0.6
            },
            'hard_skills': {
                'python': 0.9,
                'java': 0.8,
                'c++': 0.8,
                'javascript': 0.7,
                'html': 0.5,
                'css': 0.5,
                'sql': 0.7,
                'react': 0.7,
                'angular': 0.7,
                'vue': 0.7,
                'node.js': 0.7,
                'django': 0.8,
                'flask': 0.8,
                'aws': 0.8,
                'azure': 0.7,
                'gcp': 0.7,
                'docker': 0.8,
                'kubernetes': 0.8,
                'excel': 0.5,
                'powerbi': 0.6,
                'tableau': 0.6,
                'r': 0.7,
                'matlab': 0.6,
                'tensorflow': 0.9,
                'pytorch': 0.9,
                'data analysis': 0.8,
                'machine learning': 0.9,
                'ai': 0.9,
                'nlp': 0.9,
                'computer vision': 0.9,
                'devops': 0.8,
                'ci/cd': 0.7,
                'git': 0.6
            }
        }
        
        self.education_weights = {
            'high school': 0.5,
            'associate\'s degree': 0.6,
            'bachelor\'s degree': 0.8,
            'master\'s degree': 0.9,
            'phd': 1.0,
            'doctorate': 1.0,
            'mba': 0.9,
            'college graduate': 0.8,
            'undergraduate': 0.7,
            'graduate': 0.9,
            'postgraduate': 0.9
        }
        
        self.course_weights = {
            'computer science': 0.9,
            'information technology': 0.8,
            'software engineering': 0.9,
            'data science': 0.9,
            'artificial intelligence': 0.9,
            'business administration': 0.6,
            'marketing': 0.5,
            'finance': 0.6,
            'accounting': 0.5,
            'economics': 0.6,
            'engineering': 0.7,
            'mechanical engineering': 0.6,
            'electrical engineering': 0.7,
            'civil engineering': 0.6,
            'psychology': 0.5,
            'biology': 0.5,
            'chemistry': 0.5,
            'physics': 0.6,
            'mathematics': 0.7,
            'statistics': 0.8
        }
    
    def assess_candidate(self, entities):
        scores = {
            'soft_skills': 0,
            'hard_skills': 0,
            'education': 0,
            'experience': 0,
            'certification': 0
        }
        
        # Assess soft skills
        if entities['soft_skills']:
            for skill in entities['soft_skills']:
                skill_lower = skill.lower()
                if skill_lower in self.skill_weights['soft_skills']:
                    scores['soft_skills'] += self.skill_weights['soft_skills'][skill_lower]
            scores['soft_skills'] /= max(1, len(entities['soft_skills']))
        
        # Assess hard skills
        if entities['hard_skills']:
            for skill in entities['hard_skills']:
                skill_lower = skill.lower()
                if skill_lower in self.skill_weights['hard_skills']:
                    scores['hard_skills'] += self.skill_weights['hard_skills'][skill_lower]
            scores['hard_skills'] /= max(1, len(entities['hard_skills']))
        
        # Assess education
        if entities['education_level']:
            for edu in entities['education_level']:
                edu_lower = edu.lower()
                if edu_lower in self.education_weights:
                    scores['education'] += self.education_weights[edu_lower]
            scores['education'] /= max(1, len(entities['education_level']))
        
        # Assess course
        if entities['course']:
            for course in entities['course']:
                course_lower = course.lower()
                if course_lower in self.course_weights:
                    scores['education'] += self.course_weights[course_lower]
            scores['education'] /= max(2, len(entities['education_level']) + len(entities['course']))
        
        # Assess experience
        if entities['experience']:
            years_pattern = r'(\d+)'
            total_years = 0
            for exp in entities['experience']:
                match = re.search(years_pattern, exp)
                if match:
                    years = int(match.group(1))
                    total_years += years
            
            # Normalize experience score (assuming max 10 years is optimal)
            scores['experience'] = min(1.0, total_years / 10.0)
        
        # Assess certification
        if entities['certification']:
            # Simple scoring: more certifications = higher score (max 1.0)
            scores['certification'] = min(1.0, len(entities['certification']) / 3.0)
        
        # Calculate overall score (weighted average)
        weights = {
            'soft_skills': 0.2,
            'hard_skills': 0.3,
            'education': 0.2,
            'experience': 0.2,
            'certification': 0.1
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights)
        
        # Generate assessment result
        assessment = {
            'overall_score': round(overall_score * 100, 2),
            'category_scores': {key: round(scores[key] * 100, 2) for key in scores},
            'suitability': self._get_suitability_level(overall_score),
            'strengths': self._get_strengths(scores),
            'areas_for_improvement': self._get_areas_for_improvement(scores),
            'recommendation': self._get_recommendation(overall_score)
        }
        
        return assessment
    
    def _get_suitability_level(self, score):
        if score >= 0.85:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Average"
        else:
            return "Below Average"
    
    def _get_strengths(self, scores):
        strengths = []
        for category, score in scores.items():
            if score >= 0.7:
                strengths.append(category.replace('_', ' ').title())
        return strengths if strengths else ["None identified"]
    
    def _get_areas_for_improvement(self, scores):
        areas = []
        for category, score in scores.items():
            if score < 0.6:
                areas.append(category.replace('_', ' ').title())
        return areas if areas else ["None identified"]
    
    def _get_recommendation(self, score):
        if score >= 0.85:
            return "Highly recommended for the position."
        elif score >= 0.7:
            return "Recommended for the position."
        elif score >= 0.5:
            return "May be considered for the position with some reservations."
        else:
            return "Not recommended for the position at this time."

# Initialize the assessment model
assessment_model = CandidateAssessmentModel()

# Function to extract text from resume file
def extract_text_from_resume(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        text = ''
    
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    if 'resumes[]' not in request.files:
        return jsonify({'error': 'No file part'})
    
    files = request.files.getlist('resumes[]')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'})
    
    results = []
    
    for file in files:
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Extract text from resume
            resume_text = extract_text_from_resume(file_path)
            
            # Extract entities using NER model
            entities = ner_model.extract_entities(resume_text)
            
            # Add the assessment directly
            assessment = assessment_model.assess_candidate(entities)
            
            results.append({
                'filename': file.filename,
                'entities': entities,
                'assessment': assessment
            })
    
    return jsonify({
        'status': 'success',
        'results': results
    })

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    results = data.get('results', [])
    
    if not results:
        return jsonify({'error': 'No results to compare'})
    
    # Extract assessments from results
    assessments = [result.get('assessment', {}) for result in results]
    filenames = [result.get('filename', f"Candidate {i+1}") for i, result in enumerate(results)]
    
    # Prepare comparison data
    comparison = {
        'candidates': filenames,
        'overall_scores': [assessment.get('overall_score', 0) for assessment in assessments],
        'category_scores': {
            'soft_skills': [assessment.get('category_scores', {}).get('soft_skills', 0) for assessment in assessments],
            'hard_skills': [assessment.get('category_scores', {}).get('hard_skills', 0) for assessment in assessments],
            'education': [assessment.get('category_scores', {}).get('education', 0) for assessment in assessments],
            'experience': [assessment.get('category_scores', {}).get('experience', 0) for assessment in assessments],
            'certification': [assessment.get('category_scores', {}).get('certification', 0) for assessment in assessments]
        },
        'suitability': [assessment.get('suitability', 'N/A') for assessment in assessments],
        'recommendations': [assessment.get('recommendation', 'N/A') for assessment in assessments]
    }
    
    # Find the best candidate based on overall score
    best_candidate_index = comparison['overall_scores'].index(max(comparison['overall_scores']))
    comparison['best_candidate'] = {
        'index': best_candidate_index,
        'name': filenames[best_candidate_index],
        'score': comparison['overall_scores'][best_candidate_index]
    }
    
    return jsonify({
        'status': 'success',
        'comparison': comparison
    })

if __name__ == '__main__':
    app.run(debug=True)
