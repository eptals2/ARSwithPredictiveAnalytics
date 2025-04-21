import os
import textract
import PyPDF2
import docx
import pytesseract
from PIL import Image
import pandas as pd

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
        text = pytesseract.image_to_string(image, config="--psm 3")
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

# def extract_images_to_csv(folder_path, output_csv="image_resumes.csv"):
#     data = []
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             print(f"🖼️ Extracting: {filename}")
#             text = extract_text_from_image(file_path)
#             data.append({"filename": filename, "resume_text": text})

#     df = pd.DataFrame(data)
#     df.to_csv(output_csv, index=False, encoding='utf-8')
#     print(f"✅ All image resumes saved to: {output_csv}")

# # 🔽 Use your folder path
# if __name__ == "__main__":
#     folder = "C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Data-Training/resumes from PECIT"
#     extract_images_to_csv(folder)

# if __name__ == "__main__":
#     file_path = "C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Data-Training/all resumes/Accounting Profesional.pdf"
#     extracted = extract_text(file_path)
#     print(extracted)

