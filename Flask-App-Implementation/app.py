from flask import Flask, render_template, request, jsonify, redirect, session, url_for
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
import secrets


# Download necessary NLTK data
# nltk.download('punkt', quiet=True)

app = Flask(__name__)

def generate_secret_key():
    """Generate a secret key using the secrets module."""
    return secrets.token_urlsafe(16)

app.config['SECRET_KEY'] = generate_secret_key()

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
            'CERTIFICATION': r'\b(?:certification[s]?[:\s]*([\w\s]+)|certified\s+([\w\s]+)|([\w\s]+)\s+certified|([A-Z]+)(?:\s*[-–]\s*|\s+)(?:certification|certified|certificate))\b',
            'JOB_POSITION': r'\b(?:Software Engineer|Data Scientist|Full Stack Developer|Frontend Developer|Backend Developer|DevOps Engineer|System Administrator|Project Manager|Product Manager|Business Analyst|Data Analyst|Machine Learning Engineer|AI Engineer|Cloud Engineer|Network Engineer|Security Engineer|QA Engineer|UI/UX Designer|Database Administrator|IT Support|Technical Lead|Team Lead|Engineering Manager|CTO|CEO|CFO|COO|Director|VP|Senior|Junior|Mid-level|Principal|Staff|Lead)\b'
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
            'certification': [],
            'job_position': []
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
        
        # Extract job position
        job_position_matches = re.finditer(self.entity_patterns['JOB_POSITION'], text, re.IGNORECASE)
        for match in job_position_matches:
            entities['job_position'].append(match.group(0))
        
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
        
        # Job role categories based on skills and education
        self.job_categories = {
            'software_developer': {
                'skills': ['python', 'java', 'c++', 'javascript', 'html', 'css', 'sql', 'git'],
                'courses': ['computer science', 'software engineering', 'information technology'],
                'title': 'Software Developer'
            },
            'web_developer': {
                'skills': ['javascript', 'html', 'css', 'react', 'angular', 'vue', 'node.js'],
                'courses': ['computer science', 'web development', 'information technology'],
                'title': 'Web Developer'
            },
            'data_scientist': {
                'skills': ['python', 'r', 'sql', 'machine learning', 'data analysis', 'tensorflow', 'pytorch', 'statistics'],
                'courses': ['data science', 'computer science', 'statistics', 'mathematics'],
                'title': 'Data Scientist'
            },
            'data_analyst': {
                'skills': ['sql', 'python', 'r', 'excel', 'powerbi', 'tableau', 'data analysis'],
                'courses': ['data science', 'statistics', 'business administration', 'economics'],
                'title': 'Data Analyst'
            },
            'machine_learning_engineer': {
                'skills': ['python', 'tensorflow', 'pytorch', 'machine learning', 'ai', 'nlp', 'computer vision'],
                'courses': ['computer science', 'artificial intelligence', 'data science', 'mathematics'],
                'title': 'Machine Learning Engineer'
            },
            'devops_engineer': {
                'skills': ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'git', 'devops'],
                'courses': ['computer science', 'information technology', 'software engineering'],
                'title': 'DevOps Engineer'
            },
            'frontend_developer': {
                'skills': ['javascript', 'html', 'css', 'react', 'angular', 'vue'],
                'courses': ['computer science', 'web development', 'information technology'],
                'title': 'Frontend Developer'
            },
            'backend_developer': {
                'skills': ['python', 'java', 'c++', 'sql', 'django', 'flask', 'node.js'],
                'courses': ['computer science', 'software engineering', 'information technology'],
                'title': 'Backend Developer'
            },
            'fullstack_developer': {
                'skills': ['javascript', 'html', 'css', 'react', 'angular', 'vue', 'python', 'java', 'sql', 'django', 'flask', 'node.js'],
                'courses': ['computer science', 'software engineering', 'information technology'],
                'title': 'Full Stack Developer'
            },
            'database_administrator': {
                'skills': ['sql', 'database', 'oracle', 'mysql', 'postgresql', 'mongodb'],
                'courses': ['computer science', 'information technology', 'database management'],
                'title': 'Database Administrator'
            },
            'project_manager': {
                'skills': ['leadership', 'communication', 'teamwork', 'time management', 'project management'],
                'courses': ['business administration', 'project management', 'computer science'],
                'title': 'Project Manager'
            },
            'business_analyst': {
                'skills': ['communication', 'critical thinking', 'sql', 'excel', 'powerbi', 'tableau'],
                'courses': ['business administration', 'economics', 'finance', 'information technology'],
                'title': 'Business Analyst'
            }
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
        
        # Determine the best job match based on skills and education
        job_match = self._determine_job_match(entities)
        
        # Generate assessment result
        assessment = {
            'overall_score': round(overall_score * 100, 2),
            'category_scores': {key: round(scores[key] * 100, 2) for key in scores},
            'suitability': self._get_suitability_level(overall_score),
            'strengths': self._get_strengths(scores),
            'areas_for_improvement': self._get_areas_for_improvement(scores),
            'recommendation': self._get_recommendation(overall_score, job_match),
            'best_job_match': job_match
        }
        
        return assessment
    
    def _determine_job_match(self, entities):
        # Extract all skills (both hard and soft) from the entities
        all_skills = []
        if entities['hard_skills']:
            all_skills.extend([skill.lower() for skill in entities['hard_skills']])
        if entities['soft_skills']:
            all_skills.extend([skill.lower() for skill in entities['soft_skills']])
        
        # Extract all courses from the entities
        all_courses = []
        if entities['course']:
            all_courses.extend([course.lower() for course in entities['course']])
        
        # Calculate match score for each job category
        job_scores = {}
        for job_key, job_info in self.job_categories.items():
            skill_match = sum(1 for skill in all_skills if skill in job_info['skills'])
            course_match = sum(1 for course in all_courses if course in job_info['courses'])
            
            # Calculate weighted score (skills are more important than courses)
            if job_info['skills']:
                skill_score = skill_match / len(job_info['skills'])
            else:
                skill_score = 0
                
            if job_info['courses'] and all_courses:
                course_score = course_match / len(job_info['courses'])
            else:
                course_score = 0
                
            job_scores[job_key] = (skill_score * 0.7) + (course_score * 0.3)
        
        # Find the job with the highest match score
        if job_scores:
            best_job_key = max(job_scores, key=job_scores.get)
            best_job_score = job_scores[best_job_key]
            
            # Return the best match regardless of score threshold
            return {
                'title': self.job_categories[best_job_key]['title'],
                'match_score': round(best_job_score * 100, 2)
            }
        
        # If no skills or courses match, determine a job based on experience or education
        # This ensures we always return a specific job title
        fallback_jobs = self._determine_fallback_job(entities)
        return fallback_jobs
    
    def _determine_fallback_job(self, entities):
        """Determine a job based on experience or education when no good skill match is found"""
        # Check for keywords in experience
        experience_keywords = {
            'developer': 'Software Developer',
            'engineer': 'Software Engineer',
            'analyst': 'Business Analyst',
            'manager': 'Project Manager',
            'designer': 'Graphic Designer',
            'assistant': 'Administrative Assistant',
            'marketing': 'Marketing Specialist',
            'sales': 'Sales Representative',
            'customer': 'Customer Service Representative',
            'support': 'Technical Support Specialist',
            'teacher': 'Teacher',
            'writer': 'Content Writer',
            'editor': 'Editor',
            'accountant': 'Accountant',
            'finance': 'Financial Analyst',
            'hr': 'HR Specialist',
            'human resources': 'HR Specialist',
            'recruiter': 'Recruiter',
            'coordinator': 'Project Coordinator',
            'administrator': 'System Administrator',
            'web': 'Web Developer',
            'data': 'Data Analyst',
            'science': 'Data Scientist',
            'research': 'Research Assistant',
            'frontend': 'Frontend Developer',
            'backend': 'Backend Developer',
            'full stack': 'Full Stack Developer',
            'ui': 'UI Designer',
            'ux': 'UX Designer',
            'graphic': 'Graphic Designer',
            'content': 'Content Creator',
            'social media': 'Social Media Manager',
            'qa': 'QA Engineer',
            'quality': 'QA Engineer',
            'test': 'QA Engineer',
            'devops': 'DevOps Engineer',
            'cloud': 'Cloud Engineer',
            'security': 'Security Analyst',
            'network': 'Network Administrator',
            'database': 'Database Administrator',
            'product': 'Product Manager',
            'scrum': 'Scrum Master',
            'agile': 'Agile Coach',
            'mobile': 'Mobile Developer',
            'android': 'Android Developer',
            'ios': 'iOS Developer',
            'machine learning': 'Machine Learning Engineer',
            'ai': 'AI Specialist'
        }
        
        if entities['experience']:
            exp_text = ' '.join(entities['experience']).lower()
            for keyword, job_title in experience_keywords.items():
                if keyword in exp_text:
                    return {
                        'title': job_title,
                        'match_score': 75
                    }
        
        # Check for keywords in education/course
        education_keywords = {
            'computer science': 'Software Developer',
            'information technology': 'IT Specialist',
            'software engineering': 'Software Engineer',
            'data science': 'Data Scientist',
            'business': 'Business Analyst',
            'marketing': 'Marketing Specialist',
            'finance': 'Financial Analyst',
            'accounting': 'Accountant',
            'human resources': 'HR Specialist',
            'psychology': 'HR Assistant',
            'design': 'Graphic Designer',
            'art': 'Graphic Designer',
            'communication': 'Communications Specialist',
            'journalism': 'Content Writer',
            'engineering': 'Engineer',
            'education': 'Teacher',
            'nursing': 'Nurse',
            'medicine': 'Medical Professional',
            'law': 'Legal Assistant',
            'political science': 'Policy Analyst',
            'international relations': 'International Relations Specialist',
            'mathematics': 'Data Analyst',
            'statistics': 'Statistician',
            'physics': 'Research Scientist',
            'chemistry': 'Research Scientist',
            'biology': 'Research Assistant',
            'economics': 'Economic Analyst',
            'management': 'Manager',
            'hospitality': 'Hospitality Specialist',
            'tourism': 'Tourism Coordinator',
            'agriculture': 'Agricultural Specialist',
            'environmental': 'Environmental Specialist',
            'architecture': 'Architect',
            'civil': 'Civil Engineer',
            'mechanical': 'Mechanical Engineer',
            'electrical': 'Electrical Engineer',
            'web development': 'Web Developer',
            'artificial intelligence': 'AI Engineer',
            'machine learning': 'Machine Learning Engineer',
            'cybersecurity': 'Cybersecurity Analyst',
            'network': 'Network Administrator',
            'database': 'Database Administrator',
            'cloud computing': 'Cloud Engineer',
            'mobile development': 'Mobile Developer',
            'game development': 'Game Developer',
            'multimedia': 'Multimedia Designer',
            'animation': 'Animator',
            'film': 'Video Editor',
            'music': 'Music Producer',
            'photography': 'Photographer'
        }
        
        if entities['course']:
            course_text = ' '.join(entities['course']).lower()
            for keyword, job_title in education_keywords.items():
                if keyword in course_text:
                    return {
                        'title': job_title,
                        'match_score': 70
                    }
        
        # If no experience or education keywords match, check for hard skills
        if entities['hard_skills']:
            skills_text = ' '.join(entities['hard_skills']).lower()
            for keyword, job_title in experience_keywords.items():
                if keyword in skills_text:
                    return {
                        'title': job_title,
                        'match_score': 65
                    }
        
        # Default fallback based on education level
        if entities['education_level']:
            edu_level = ' '.join(entities['education_level']).lower()
            if 'master' in edu_level or 'phd' in edu_level or 'doctorate' in edu_level:
                return {
                    'title': 'Senior Specialist',
                    'match_score': 60
                }
            elif 'bachelor' in edu_level or 'degree' in edu_level:
                return {
                    'title': 'Professional Specialist',
                    'match_score': 60
                }
            elif 'associate' in edu_level or 'diploma' in edu_level:
                return {
                    'title': 'Technical Specialist',
                    'match_score': 60
                }
        
        # Final fallback options
        return {
            'title': 'Professional Specialist',
            'match_score': 50
        }
    
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
    
    def _get_recommendation(self, score, job_match):
        job_title = job_match['title']
        
        if score >= 0.85:
            return f"Highly recommended for the {job_title} position. The candidate's skills and experience are an excellent match for this role."
        elif score >= 0.7:
            return f"Recommended for the {job_title} position. The candidate has a good set of skills and qualifications for this role."
        elif score >= 0.5:
            return f"May be considered for the {job_title} position with some reservations. Additional training or experience may be beneficial."
        else:
            # For low scores, suggest alternative positions that might be a better fit
            alternative_positions = self._suggest_alternative_positions(job_match['title'])
            if alternative_positions:
                return f"Not recommended for the {job_title} position at this time. Based on the candidate's profile, they might be better suited for: {', '.join(alternative_positions)}."
            else:
                return f"Not recommended for the {job_title} position at this time. The candidate may benefit from additional training and experience."
    
    def _suggest_alternative_positions(self, current_job):
        # Map of job titles to potential alternative positions
        job_alternatives = {
            'Software Developer': ['Junior Developer', 'QA Engineer', 'Technical Support Specialist'],
            'Web Developer': ['Web Designer', 'UI/UX Designer', 'Content Manager'],
            'Data Scientist': ['Data Analyst', 'Business Analyst', 'Research Assistant'],
            'Data Analyst': ['Business Intelligence Analyst', 'Market Research Analyst', 'Junior Data Scientist'],
            'Machine Learning Engineer': ['Data Scientist', 'Software Developer', 'Research Assistant'],
            'DevOps Engineer': ['System Administrator', 'IT Support Specialist', 'Cloud Engineer'],
            'Frontend Developer': ['Web Designer', 'UI/UX Designer', 'HTML/CSS Developer'],
            'Backend Developer': ['Database Administrator', 'API Developer', 'Junior Software Developer'],
            'Full Stack Developer': ['Frontend Developer', 'Backend Developer', 'Web Developer'],
            'Database Administrator': ['Data Analyst', 'IT Support Specialist', 'System Administrator'],
            'Project Manager': ['Team Lead', 'Product Owner', 'Scrum Master'],
            'Business Analyst': ['Data Analyst', 'Market Research Analyst', 'Project Coordinator'],
            'Entry-level Position': ['Intern', 'Assistant', 'Junior Specialist']
        }
        
        return job_alternatives.get(current_job, ['Entry-level Position', 'Intern', 'Assistant'])

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
    # elif file_extension == '.txt':
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         text = file.read()
    else:
        text = ''
    
    return text

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

@app.route('/rank', methods=['POST'])
def rank():
    try:
        data = request.get_json()
        results = data.get('results', [])
        
        # Sort results by overall score in descending order
        sorted_results = sorted(results, key=lambda x: x['assessment']['overall_score'], reverse=True)
        
        # Get the best candidate
        best_candidate = {
            'name': sorted_results[0]['filename'],
            'score': sorted_results[0]['assessment']['overall_score'],
            'index': 0
        }
        
        return jsonify({
            'status': 'success',
            'comparison': {
                'candidates': [result['filename'] for result in sorted_results],
                'overall_scores': [result['assessment']['overall_score'] for result in sorted_results],
                'best_candidate': best_candidate
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
