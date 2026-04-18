"""Module for job suitability prediction using Flask and machine learning."""

import os
import textract
import PyPDF2
import docx
import pytesseract
from PIL import Image

# ✅ Set Tesseract OCR Path (Windows Only) - Ensure it's installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_docx(file):
    """Extract text from DOCX files."""
    try:
        doc = docx.Document(file)
        return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"❌ Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_pdf(file):
    """Extract text from PDFs using PyPDF2."""
    try:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text.strip() if text else ""
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""

def extract_text_from_image(file):
    """Extract text from images using Tesseract OCR."""
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image, config="--psm 3")  # Auto mode
        return text.strip() if text else ""
    except Exception as e:
        print(f"❌ Error extracting text from image: {e}")
        return ""

def extract_text(file):
    """Detect file type and extract text accordingly, supporting Flask's FileStorage objects."""
    
    if hasattr(file, "filename"):  
        filename = file.filename
    else:
        filename = file  # If it's already a string path
    
    _, ext = os.path.splitext(filename)  # Get file extension
    ext = ext.lower()

    text = ""

    if ext == ".docx":
        text = extract_text_from_docx(file)
    elif ext == ".pdf":
        text = extract_text_from_pdf(file)
    elif ext in (".jpg", ".jpeg", ".png"):
        text = extract_text_from_image(file)

    # 🔄 Fallback to Textract if primary extraction fails
    if not text:
        try:
            text = textract.process(file).decode("utf-8").strip()
        except Exception as e:
            print(f"❌ Error extracting text using textract: {e}")
            return ""

    return text
