from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split

# Define entity labels based on BIO scheme
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

def prepare_dataset(df):
    """Convert DataFrame to HuggingFace Dataset format"""
    dataset_dict = {
        'tokens': df['tokens'].tolist(),
        'ner_tags': df['ner_tags'].tolist()
    }
    return Dataset.from_dict(dataset_dict)

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

def train_roberta_ner():
    # Load tokenizer and model
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForTokenClassification.from_pretrained(
        'roberta-base',
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    # Load and prepare training data
    df = pd.read_csv('ner_training_data.csv')
    
    # Convert string representations to lists
    df['tokens'] = df['tokens'].apply(eval)
    df['ner_tags'] = df['ner_tags'].apply(eval)
    
    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = prepare_dataset(train_df)
    val_dataset = prepare_dataset(val_df)
    
    # Tokenize datasets
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
        save_strategy="epoch"
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
    trainer.train()
    
    # Save model
    trainer.save_model("models/RoBERTa-fine-tuned-model")
    tokenizer.save_pretrained("models/RoBERTa-fine-tuned-model")

if __name__ == "__main__":
    train_roberta_ner()
