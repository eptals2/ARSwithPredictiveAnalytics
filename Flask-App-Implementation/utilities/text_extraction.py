import os
import tempfile
import werkzeug
from werkzeug.utils import secure_filename
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
        print(f"❌ Error extracting text from DOCX: {e}")
        return None

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text if text else None
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return None

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, config="--psm 6")
        return text.strip() if text else None
    except Exception as e:
        print(f"❌ Error extracting text from image: {e}")
        return None

def extract_text(file):
    """Extract text from an uploaded file (Flask FileStorage or file path)."""

    # ✅ Handle Flask FileStorage input
    if isinstance(file, werkzeug.datastructures.FileStorage):
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)  # Save file temporarily
        file_path = temp_path
    elif isinstance(file, str):  # If a file path is provided
        file_path = file
    else:
        print(f"🔥 ERROR: Unsupported file type! Received: {type(file)} → {file}")
        return None

    text = None
    file_ext = file_path.lower()

    if file_ext.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif file_ext.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_ext.endswith((".jpg", ".jpeg", ".png")):
        text = extract_text_from_image(file_path)

    # 🔄 Fallback to Textract if primary extraction fails
    if text is None:
        try:
            text = textract.process(file_path).decode("utf-8")
        except Exception as e:
            print(f"❌ Textract failed: {e}")

    # ✅ Cleanup (only for Flask uploads)
    if isinstance(file, werkzeug.datastructures.FileStorage):
        os.remove(file_path)

    return text
