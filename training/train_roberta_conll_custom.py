from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split

# Define entity labels based on memory specifications
LABELS = [
    "O",  # Outside of named entity
    "B-AGE", "I-AGE",
    "B-GENDER", "I-GENDER",
    "B-ADDRESS", "I-ADDRESS",
    "B-SKILLS", "I-SKILLS",
    "B-EXPERIENCE", "I-EXPERIENCE",
    "B-EDUCATION", "I-EDUCATION",
    "B-CERTIFICATION", "I-CERTIFICATION"
]

# Create label mapping
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

def read_custom_conll_file(file_path):
    """Read CoNLL format file with custom format: token -X- _ label"""
    current_tokens = []
    current_labels = []
    all_tokens = []
    all_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line:
                # Split by spaces and get token and label
                parts = line.split()
                if len(parts) >= 4:  # Ensure we have all parts: token -X- _ label
                    token = parts[0]
                    label = parts[3]  # The label is the fourth part
                    current_tokens.append(token)
                    current_labels.append(label)
            elif current_tokens:  # Empty line indicates end of sentence
                all_tokens.append(current_tokens)
                all_labels.append(current_labels)
                current_tokens = []
                current_labels = []
    
    # Add the last sentence if file doesn't end with empty line
    if current_tokens:
        all_tokens.append(current_tokens)
        all_labels.append(current_labels)
    
    # Convert labels to ids and handle unknown labels
    processed_labels = []
    for labels in all_labels:
        label_ids = []
        for label in labels:
            if label in label2id:
                label_ids.append(label2id[label])
            else:
                # Handle unknown labels as O (Outside)
                label_ids.append(label2id["O"])
        processed_labels.append(label_ids)
    
    return pd.DataFrame({
        'tokens': all_tokens,
        'ner_tags': processed_labels
    })

def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize text and align labels with tokens"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
        padding="max_length"
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def prepare_dataset(df):
    """Convert DataFrame to HuggingFace Dataset format"""
    return Dataset.from_dict({
        'tokens': df['tokens'].tolist(),
        'ner_tags': df['ner_tags'].tolist()
    })

def train_roberta_ner(conll_file_path):
    """Train RoBERTa NER model using custom CoNLL format data"""
    # Load tokenizer and model
    print("Loading RoBERTa model and tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForTokenClassification.from_pretrained(
        'roberta-base',
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    # Load CoNLL data
    print("Loading custom CoNLL data...")
    df = read_custom_conll_file(conll_file_path)
    
    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_df)
    val_dataset = prepare_dataset(val_df)
    
    # Tokenize datasets
    print("Tokenizing data...")
    train_tokenized = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/RoBERTa-fine-tuned-model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model("models/RoBERTa-fine-tuned-model")
    tokenizer.save_pretrained("models/RoBERTa-fine-tuned-model")
    print("Training complete!")

if __name__ == "__main__":
    # Specify your CoNLL file path
    conll_file_path = "path/to/your/label_studio_export.conll"
    train_roberta_ner(conll_file_path)
