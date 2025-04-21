import os
import textract
import PyPDF2
import docx
import pandas as pd
import pytesseract
from PIL import Image
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import inflect
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Download necessary nltk data (if needed)
# nltk.download('stopwords')
# nltk.download('wordnet')

# Set up inflect engine for number-to-word conversion
p = inflect.engine()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Set Tesseract OCR Path (Windows Only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load RoBERTa model for entity extraction (if available)
try:
    tokenizer = AutoTokenizer.from_pretrained("models/RoBERTa-fine-tuned-model")
    model = AutoModelForTokenClassification.from_pretrained("models/RoBERTa-fine-tuned-model")
    ROBERTA_MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load RoBERTa model: {e}")
    print("Falling back to basic text processing")
    ROBERTA_MODEL_LOADED = False

#----------------------------
# TEXT EXTRACTION FUNCTIONS
#----------------------------
def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from {file_path} using python-docx: {e}")
        return None

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text if text else None
    except Exception as e:
        print(f"Error extracting text from {file_path} using PyPDF2: {e}")
        return None

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, config='--psm 6')
        return text.strip() if text else None
    except Exception as e:
        print(f"Error extracting text from {file_path} using OCR: {e}")
        return None

def extract_text(file_path):
    """Detect file type and extract text accordingly."""
    text = None
    if file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith((".jpg", ".jpeg", ".png")):
        text = extract_text_from_image(file_path)
    
    # Fallback to Textract if primary extraction fails
    if text is None:
        try:
            text = textract.process(file_path).decode("utf-8")
        except Exception as e:
            print(f"Error extracting text from {file_path} using textract: {e}")
            return None
    
    return text

#----------------------------
# TEXT PREPROCESSING
#----------------------------
def preprocess_text(text):
    """Lowercase, remove special characters, normalize numbers, lemmatize, and remove stopwords."""
    if not text:
        return ""

    text = text.lower()
    
    # Normalize numbers (convert digits to words)
    words = text.split()
    normalized_words = []
    for word in words:
        if word.isdigit():  # Check if it's a number
            try:
                word = p.number_to_words(int(word))  # Convert to words
            except:
                pass  # If conversion fails, keep the original number
        normalized_words.append(word)
    
    text = " ".join(normalized_words)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Lemmatization and Stopword Removal
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

    return text

#----------------------------
# ENTITY EXTRACTION
#----------------------------
def extract_entities(text):
    """
    Extract entities from text using RoBERTa model or fallback to regex patterns.
    Returns a dictionary of entity types and their values.
    """
    entities = {
        "age": "none",
        "gender": "none",
        "address": "none",
        "skills": [],
        "experience": [],
        "education": [],
        "certification": []
    }
    
    if ROBERTA_MODEL_LOADED:
        # Use RoBERTa model for entity extraction
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process outputs and populate entities dictionary
            # This is a placeholder - actual implementation would depend on model output format
            # ...
            
            return entities
        except Exception as e:
            print(f"Error using RoBERTa for entity extraction: {e}")
            print("Falling back to regex patterns")
    
    # Fallback to basic regex patterns for entity extraction
    # Age extraction
    age_pattern = r'\b(?:age|years old|aged)\s*:?\s*(\d+)\b'
    age_match = re.search(age_pattern, text, re.IGNORECASE)
    if age_match:
        entities["age"] = age_match.group(1)
    
    # Gender extraction
    if re.search(r'\b(?:male|man|gentleman|boy)\b', text, re.IGNORECASE):
        entities["gender"] = "male"
    elif re.search(r'\b(?:female|woman|lady|girl)\b', text, re.IGNORECASE):
        entities["gender"] = "female"
    
    # Address extraction (simplified)
    address_pattern = r'\b(?:address|location|residence)\s*:?\s*([^.]*)'
    address_match = re.search(address_pattern, text, re.IGNORECASE)
    if address_match:
        entities["address"] = address_match.group(1).strip()
    
    # Skills extraction (simplified)
    skill_keywords = ["python", "java", "javascript", "c++", "sql", "html", "css", 
                      "web development", "programming", "software", "data analysis", 
                      "machine learning", "data science", "data engineer", "data analyst", 
                      "data visualization", "data mining", "data wrangling", "data cleaning",
                      "data science", "data engineer", "data analyst", "data visualization", 
                      "data mining", "data wrangling", "data cleaning", "business intelligence", 
                      "cashier", "cash handling", "point of sale", "pos", "customer service", 
                      "inventory management", "cash register", "money handling", "retail", 
                      "transaction processing", "balancing cash drawer", "payment processing",
                      "sales", "upselling", "cross-selling", "merchandising", "product knowledge",
                      "driver", "driving", "cdl", "chauffeur", "delivery", "trucking", "transportation", 
                      "fleet management", "logistics", "vehicle operation", "defensive driving", "route planning",
                      "commercial driver", "driver's license", "forklift", "heavy equipment", "truck driver",
                      "courier", "dispatch", "road safety", "transportation logistics", "vehicle maintenance",
                      "filipino teacher", "college instructor", "music teacher", "psychologist", 
                      "english teacher", "pe teacher", "physical education", "full-stack", 
                      "web developer", "computer engineer", "hr", "human resources", 
                      "office staff", "administrative", "nurse", "nursing", "healthcare", 
                      "tle teacher", "math teacher", "mathematics", "elementary teacher", 
                      "criminologist", "criminology", "social studies teacher", "history teacher",
                      "education", "teaching", "instruction", "classroom management", 
                      "lesson planning", "curriculum development", "student assessment",
                      "academic advising", "mentoring", "tutoring", "grading", 
                      "educational technology", "learning management systems", "lms",
                    "call handling", "customer service", "phone etiquette", "problem solving", 
                    "conflict resolution", "active listening", "communication", "patience", 
                    "empathy", "multitasking", "data entry", "crm software", "technical support", 
                    "sales", "upselling", "product knowledge", "typing", "computer literacy", 
                    "time management", "stress management", "call center", "telemarketing", 
                    "cold calling", "customer retention", "quality assurance",
                    "computer technician", "hardware troubleshooting", "system maintenance", 
                    "network setup", "software installation", "pc repair", "it support", 
                    "diagnostics", "computer assembly", "virus removal", "data recovery", 
                    "windows administration", "bios configuration", "device drivers", 
                    "system upgrades", "preventative maintenance", "technical documentation", 
                    "remote assistance", "desktop support", "hardware replacement",
                    "office administration", "filing", "record keeping", "clerical", "data entry",
                    "microsoft office", "word processing", "spreadsheets", "scheduling",
                    "calendar management", "reception", "phone handling", "correspondence",
                    "document preparation", "office management", "bookkeeping", "administrative support",
                    "typing", "minute taking", "mail processing", "supplies management",
                    "organizational skills", "attention to detail", "time management",
                    "customer service", "confidentiality", "multitasking", "office equipment",
                    "photocopying", "scanning", "faxing", "email management", "office coordination",
                    ]
    for skill in skill_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            entities["skills"].append(skill.lower())
    
    # Experience extraction (simplified)
    exp_keywords = ["software development", "programming", "web development", 
                   "internship", "developer", "engineer", "consulting", "consultant",
                   "project manager", "team lead", "product owner", "scrum master",
                   "software engineer", "front-end", "back-end", "full-stack",
                   "data scientist", "data analyst", "database administrator", 
                   "system administrator", "devops", "cloud engineer", "network engineer",
                   "it specialist", "tech support", "help desk", "qa engineer",
                   "quality assurance", "tester", "ux designer", "ui developer",
                   "mobile developer", "android developer", "ios developer",
                   "machine learning engineer", "ai specialist", "business analyst",
                   "product manager", "cybersecurity analyst", "information security",
                   "solutions architect", "technical writer", "it manager",
                   # Accounting & Finance
                   "accountant", "auditor", "bookkeeper", "financial analyst", "tax preparer", 
                   "payroll specialist", "budget analyst", "treasurer", "controller", "credit analyst",
                   # Healthcare
                   "nurse", "medical assistant", "physician", "therapist", "pharmacist", 
                   "healthcare administrator", "medical technician", "radiologist", "dental hygienist",
                   # Education
                   "teacher", "professor", "tutor", "curriculum developer", "school administrator",
                   "principal", "academic advisor", "educational consultant", "instructional designer",
                   # Hospitality & Tourism
                   "hotel manager", "chef", "event planner", "tour guide", "travel agent",
                   "restaurant manager", "concierge", "flight attendant", "casino manager",
                   # Manufacturing
                   "production manager", "quality control", "machinist", "plant supervisor",
                   "industrial engineer", "assembler", "fabricator", "welder", "equipment operator",
                   # Retail & Sales
                   "sales representative", "retail manager", "merchandiser", "buyer", "store manager",
                   "sales director", "account executive", "cashier", "customer service representative",
                   # Marketing & Communications
                   "marketing manager", "public relations", "content writer", "social media manager",
                   "communications director", "brand manager", "copywriter", "media planner",
                   # Construction & Trades
                   "construction manager", "electrician", "plumber", "carpenter", "hvac technician",
                   "general contractor", "project manager", "civil engineer", "architect",
                   # Transportation & Logistics
                   "logistics coordinator", "supply chain manager", "warehouse supervisor",
                   "transportation planner", "inventory manager", "shipping coordinator", "fleet manager",
                   # Legal & Government
                   "lawyer", "paralegal", "legal assistant", "compliance officer", "government official",
                   "policy analyst", "legislative aide", "city planner", "public administrator"
                   ]
    for exp in exp_keywords:
        if re.search(r'\b' + re.escape(exp) + r'\b', text, re.IGNORECASE):
            entities["experience"].append(exp.lower())
    
    # Education extraction (simplified)
    edu_keywords = ["computer science", "information technology", "software engineering", 
                   "bachelor", "master", "phd", "degree", "mba", "doctorate", "associate", 
                   "diploma", "certificate", "engineering", "business administration", 
                   "marketing", "finance", "accounting", "economics", "psychology", 
                   "biology", "chemistry", "physics", "mathematics", "statistics", 
                   "liberal arts", "communications", "journalism", "graphic design", 
                   "fine arts", "architecture", "nursing", "healthcare administration", 
                   "public health", "medicine", "law", "criminal justice", "education", 
                   "english", "history", "political science", "international relations", 
                   "hospitality management", "culinary arts", "music", "theater", 
                   "film", "environmental science", "geology", "agriculture", "vocational course",
                   "diploma in computer science", "diploma in information technology", 
                   "diploma in software engineering", "diploma in web development",
                   "diploma in network administration", "diploma in cybersecurity",
                   "diploma in data analytics", "diploma in graphic design",
                   "diploma in business administration", "diploma in marketing",
                   "diploma in accounting", "diploma in human resources",
                   "diploma in hospitality management", "diploma in culinary arts",
                   "diploma in healthcare administration", "diploma in nursing",
                   "diploma in mechanical engineering", "diploma in electrical engineering",
                   "diploma in civil engineering", "diploma in automotive technology"]
    for edu in edu_keywords:
        if re.search(r'\b' + re.escape(edu) + r'\b', text, re.IGNORECASE):
            entities["education"].append(edu.lower())
    
    # Certification extraction (simplified)
    cert_keywords = ["microsoft", "aws", "oracle", "cisco", "comptia", "certified"]
    for cert in cert_keywords:
        if re.search(r'\b' + re.escape(cert) + r'\b', text, re.IGNORECASE):
            entities["certification"].append(cert.lower())
    
    return entities

#----------------------------
# JACCARD SIMILARITY
#----------------------------
def calculate_jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity between two sets.
    Returns normalized score (minimum 0.01 if any match exists) and intersection.
    """
    if not set1 or not set2:
        return 0, set()
        
    intersection = set1.intersection(set2)
    
    # No intersection means no similarity
    if not intersection:
        return 0, intersection
    
    union = set1.union(set2)
    
    # Calculate standard Jaccard similarity
    standard_jaccard = len(intersection) / len(union) if union else 0
    
    # Calculate weighted similarity that gives more importance to matches 
    # against the job description (typically the shorter document)
    weighted_similarity = len(intersection) / len(set2) if set2 else 0
    
    # Combine the scores with more weight given to the job-focused similarity
    combined_score = (0.3 * standard_jaccard) + (0.7 * weighted_similarity)
    
    # Ensure minimum score of 0.01 if there's any match
    normalized_score = max(0.01, combined_score)
    
    return normalized_score, intersection

#----------------------------
# ENTITY-SPECIFIC SCORING
#----------------------------
def calculate_entity_scores(resume_entities, job_entities):
    """
    Calculate Jaccard similarity scores for each entity type with appropriate weights.
    Returns a dictionary of scores and a total weighted score.
    """
    scores = {}
    
    # Age similarity (binary: 1.0 if present)
    scores["age"] = 1.0 if resume_entities["age"] != "none" else 0
    
    # Gender similarity (binary: 1.0 if present)
    scores["gender"] = 1.0 if resume_entities["gender"] != "none" else 0
    
    # Address similarity (binary: 1.0 if present)
    scores["address"] = 1.0 if resume_entities["address"] != "none" else 0
    
    # Skills similarity (Jaccard)
    resume_skills = set(resume_entities["skills"])
    job_skills = set(job_entities["skills"])
    scores["skills"], _ = calculate_jaccard_similarity(resume_skills, job_skills)
    
    # Experience similarity (Jaccard)
    resume_exp = set(resume_entities["experience"])
    job_exp = set(job_entities["experience"])
    scores["experience"], _ = calculate_jaccard_similarity(resume_exp, job_exp)
    
    # Education similarity (Jaccard)
    resume_edu = set(resume_entities["education"])
    job_edu = set(job_entities["education"])
    scores["education"], _ = calculate_jaccard_similarity(resume_edu, job_edu)
    
    # Certification similarity (Jaccard)
    resume_cert = set(resume_entities["certification"])
    job_cert = set(job_entities["certification"])
    scores["certification"], _ = calculate_jaccard_similarity(resume_cert, job_cert)
    
    # Calculate total weighted score based on diagram weights
    # Age: 0.10, Gender: 0.10, Address: 0.10, Skills: 0.20, 
    # Experience: 0.15, Education: 0.15, Certification: 0.20
    weights = {
        "age": 0.10,
        "gender": 0.10,
        "address": 0.10,
        "skills": 0.20,
        "experience": 0.15,
        "education": 0.15,
        "certification": 0.20
    }
    
    total_score = sum(scores[entity] * weights[entity] for entity in scores)
    # Convert to percentage
    total_score = total_score * 100
    
    return scores, total_score

#----------------------------
# SUITABILITY ASSESSMENT
#----------------------------
def assess_suitability(total_score):
    """
    Determine suitability based on total score thresholds from the diagram.
    """
    if total_score >= 75:
        return "Highly Suitable"
    elif total_score >= 50:
        return "Mildly Suitable"
    elif total_score >= 25:
        return "Less Suitable"
    else:
        return "Not Suitable"

#----------------------------
# PROCESS FILES & GENERATE CSV
#----------------------------
def process_files(resume_folder, job_folder, output_csv):
    data = []
    resumes = []
    job_requirements = []
    
    # Extract and preprocess resumes
    for filename in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, filename)
        if file_path.endswith((".pdf", ".docx", ".jpg", ".jpeg", ".png")):
            print(f"Extracting text from resume: {filename}")
            text = extract_text(file_path)
            if text:
                preprocessed_text = preprocess_text(text)
                resume_entities = extract_entities(text)  # Use original text for entity extraction
                resumes.append({
                    "filename": filename, 
                    "text": preprocessed_text,
                    "entities": resume_entities
                })

    # Extract and preprocess job descriptions
    for filename in os.listdir(job_folder):
        file_path = os.path.join(job_folder, filename)
        if file_path.endswith((".pdf", ".docx", ".jpg", ".jpeg", ".png")):
            print(f"Extracting text from job requirement: {filename}")
            text = extract_text(file_path)
            if text:
                preprocessed_text = preprocess_text(text)
                job_entities = extract_entities(text)  # Use original text for entity extraction
                job_requirements.append({
                    "filename": filename, 
                    "text": preprocessed_text,
                    "entities": job_entities
                })

    # Compute entity-specific scores and assess suitability
    for resume in resumes:
        for job in job_requirements:
            # Calculate overall text similarity (legacy)
            text_jaccard, common_words = calculate_jaccard_similarity(
                set(resume["text"].split()), 
                set(job["text"].split())
            )
            
            # Calculate entity-specific scores
            entity_scores, total_score = calculate_entity_scores(
                resume["entities"], 
                job["entities"]
            )
            
            # Assess suitability
            suitability = assess_suitability(total_score)
            
            # Prepare data for CSV
            row_data = {
                "Resume Filename": resume["filename"],
                "Job Filename": job["filename"],
                "Text Jaccard Score": text_jaccard,
                "Common Words": " ".join(common_words),
                "Total Score": total_score,
                "Suitability": suitability
            }
            
            # Add entity-specific scores
            for entity, score in entity_scores.items():
                row_data[f"{entity.capitalize()} Score"] = score
            
            # Add entity values
            for entity, value in resume["entities"].items():
                if isinstance(value, list):
                    row_data[f"Resume {entity.capitalize()}"] = ", ".join(value) if value else "none"
                else:
                    row_data[f"Resume {entity.capitalize()}"] = value
            
            for entity, value in job["entities"].items():
                if isinstance(value, list):
                    row_data[f"Job {entity.capitalize()}"] = ", ".join(value) if value else "none"
                else:
                    row_data[f"Job {entity.capitalize()}"] = value
            
            data.append(row_data)

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Extraction complete. Data saved to {output_csv}")

# Example usage
resume_folder = "./for-docu/all resumes"  # Change this to your resume folder path
job_folder = "./for-docu/all job requirements"  # Change this to your job requirements folder path
output_csv = "./for-docu/with-jaccard-score-dataset.csv"
process_files(resume_folder, job_folder, output_csv)
