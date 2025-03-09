import pandas as pd
import nltk

# Ensure you have the tokenizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Define the label mappings
label_map = {
    "Age": "AGE",
    "Gender": "GEN",
    "Address": "LOC",
    "Skills": "SKILL",
    "Education": "EDU",
    "Experience": "EXP",
    "Certificates": "CERT"
}

# Load the dataset /raw text
# data = """
# sample raw text here 
# """

with open('resumes_raw_text.txt', 'r') as file:
    data = file.read()

# Convert to DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data))

# Function to convert row into NER format
def convert_row_to_ner(row):
    output = []
    
    for col in df.columns:
        if pd.notna(row[col]):  # Ignore empty values
            tokens = word_tokenize(str(row[col]))  # Tokenize words
            entity_type = label_map[col]  # Get entity type

            for i, token in enumerate(tokens):
                label = f"B-{entity_type}" if i == 0 else f"I-{entity_type}"
                output.append(f"{token} {label}")
    
    return output

# Convert all rows
ner_formatted_data = []
for _, row in df.iterrows():
    ner_formatted_data.extend(convert_row_to_ner(row))
    ner_formatted_data.append("")  # Add a newline between resumes

# Save to file
with open("resume_labelled.txt", "w") as f:
    f.write("\n".join(ner_formatted_data))

# Print sample output
print("\n".join(ner_formatted_data[:20]))  # Show only first 20 lines