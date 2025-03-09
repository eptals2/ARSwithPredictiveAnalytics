import inflect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pytesseract
import docx
import PyPDF2
from PIL import Image
import textract
import re
import os

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

# Process each file in the specified directory
data_dir = "Data-Training/resumes"  # Input directory
output_dir = "Data-Training/Extracted_Text"  # Output directory

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(data_dir):
    print(f"Directory '{data_dir}' not found.")
else:
    for file in os.listdir(data_dir):
        if file.endswith((".jpeg", ".JPG", ".docx", ".pdf", ".png", ".jpg")):
            file_path = os.path.join(data_dir, file)
            text = extract_text(file_path)
            if text:
                output_file_name = os.path.splitext(file)[0] + ".txt"
                output_file_path = os.path.join(output_dir, output_file_name) #save to new folder.

                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Extracted and saved text to {output_file_name}")
            else:
                print(f"Could not extract text from {file}")