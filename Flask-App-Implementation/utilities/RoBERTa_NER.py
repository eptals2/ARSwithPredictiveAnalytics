import torch
import pandas as pd
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForTokenClassification

# Load Fine-tuned RoBERTa Model and Tokenizer
MODEL_PATH = "Flask-App-Implementation/models/RoBERTa-fine-tuned-model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForTokenClassification.from_pretrained(MODEL_PATH).to(device)

# Declare label names
label_names = ["O", "B-AGE", "I-AGE", "B-GENDER", "I-GENDER", "B-ADDRESS", "I-ADDRESS", "B-SKILLS", "I-SKILLS", "B-EXPERIENCE", "I-EXPERIENCE", "B-EDUCATION", "I-EDUCATION", "B-CERTIFICATION", "I-CERTIFICATION", "B-Others", "B-Role", "I-Others", "I-Role"]

def extract_entities(text):
    """Extract entities from resume using RoBERTa model"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits
    predictions = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    extracted_info = {"age": "", "gender": "", "address": "", "skills": "", "education": "", "experience": "", "certification": ""}
    current_entity = None
    
    for token, pred in zip(tokens, predictions[0]):
        label = label_names[pred.item()]
        if label.startswith("B-"):
            current_entity = label[2:].lower()
            extracted_info[current_entity] = token
        elif label.startswith("I-") and current_entity:
            extracted_info[current_entity] += " " + token
        else:
            current_entity = None
    
    return extracted_info