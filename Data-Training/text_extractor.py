
import inflect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pytesseract
import docx
import PyPDF2
from PIL import Image
import textract
import re

# Set up inflect engine for number-to-word conversion
p = inflect.engine()

# Set up stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Set Tesseract OCR Path (Windows Only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
    elif file_path.endswith((".jpg", ".jpeg", ".png", ".JPG")):
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
                word = p.number_to_words(int(word))
                # pass  # If conversion fails, keep the original number
        normalized_words.append(word)
    
    text = " ".join(normalized_words)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Lemmatization and Stopword Removal
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

    return text

import os

for file in os.listdir("Flask-App-Implementation/resumes"):
    if file.endswith((".jpeg",".JPG")):
        text = extract_text(os.path.join("Flask-App-Implementation/resumes", file))
        if text:
            for each_resume in text.split():
                with open('resumes_text.txt', 'a') as f:
                    f.write(each_resume + '\n')