import os
import docx
import PyPDF2
from PIL import Image
import pytesseract
import textract
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import inflect
from transformers import pipeline

# Download necessary NLTK resources (if not already downloaded)
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
p = inflect.engine()


# Set Tesseract OCR Path (Windows Only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load RoBERTa NER model
ner_pipeline = pipeline('ner', model='C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Data-Training/models/RoBERTa-fine-tuned-model',
                        tokenizer='C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Data-Training/models/RoBERTa-fine-tuned-model')

# Function to extract named entities
def extract_entities(text):
    """Extract named entities using RoBERTa NER"""
    if not text:
        return set() #return empty set if no text is given.
    ner_results = ner_pipeline(text)
    entities = set()  # Use a set to remove duplicates

    for entity in ner_results:
        if entity['score'] > 0.7:  # Filter low-confidence entities
            entities.add(entity['word'].strip())

    return entities

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

def extract_text_from_folder(folder_path):
    """Extract text from all files in a folder and return as a DataFrame."""
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            extracted_text = extract_text(file_path)
            if extracted_text is not None:
                preprocessed_text = preprocess_text(extracted_text) #preprocess the extracted text
                entities = extract_entities(preprocessed_text) #extract entities from preprocessed text.
                data.append({"filename": filename, "extracted_raw_text": extracted_text, "preprocessed_text": preprocessed_text, "entities": entities})
            else:
                data.append({"filename": filename, "extracted_raw_text": None, "preprocessed_text": None, "entities": None})
    return pd.DataFrame(data)

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

# Example usage:
folder_path = 'C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Data-Training/all resumes'  # Replace with the actual path to your folder
df = extract_text_from_folder(folder_path)

# Print the DataFrame
print(df)

#Optionally save to csv
df.to_csv('all_resumes_extractedentity.csv', index=False)