import os
import pdfplumber
from docx import Document
from tkinter import Tk, filedialog

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to select files and extract text
def process_files():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title="Select PDF or DOCX Files", filetypes=[("PDF & DOCX", "*.pdf;*.docx")])
    
    if not file_paths:
        print("No files selected.")
        return

    merged_text = ""

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        
        if file_path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            continue
        
        merged_text += f"=== {file_name} ===\n{text.strip()}\n\n"  # Add separator + newline

    # Save merged text to a single file
    output_txt_path = "merged_text.txt"
    with open(output_txt_path, "w", encoding="utf-8") as file:
        file.write(merged_text)
    
    print(f"Merged text saved to {output_txt_path}")

# Run the file processing function
process_files()
