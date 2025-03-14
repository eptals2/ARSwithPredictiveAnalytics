import os
import PyPDF2
import docx

def extract_text_from_resume(file_path):
    """Extract text from PDF and DOCX resumes"""
    file_extension = os.path.splitext(file_path)[1].lower()
    text = ''
    if file_extension == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ''
    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text.strip()