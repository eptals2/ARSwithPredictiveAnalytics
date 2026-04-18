"""Module for job suitability prediction using Flask and machine learning."""

import os
import textract
import PyPDF2
import docx
import pytesseract
from PIL import Image

# Set Tesseract OCR Path (Windows Only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
    
    # ✅ Ensure file_path is a string
    if not isinstance(file_path, str):
        print(f"🔥 ERROR: file_path is not a string! Received: {type(file_path)} → {file_path}")
        return None

    text = None
    file_path = file_path.lower()  # Normalize case

    if file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith((".jpg", ".jpeg", ".png")):
        text = extract_text_from_image(file_path)
    
    # 🔄 Fallback to Textract if primary extraction fails
    if text is None:
        try:
            text = textract.process(file_path).decode("utf-8")
        except Exception as e:
            print(f"❌ Error extracting text from {file_path} using textract: {e}")
            return None
    
    return text
