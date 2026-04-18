from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
import os

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

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def align_predictions(predictions: np.ndarray, labels: np.ndarray) -> tuple:
    """Align predictions and labels, removing special tokens."""
    preds = predictions.argmax(axis=2)
    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if labels[i, j] != -100:
                out_label_list[i].append(id2label[labels[i][j]])
                preds_list[i].append(id2label[preds[i][j]])

    return preds_list, out_label_list

def compute_metrics(eval_pred) -> Dict:
    """Compute metrics for NER evaluation."""
    predictions, labels = eval_pred
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Align predictions and labels
    preds_list, out_label_list = align_predictions(predictions, labels)

    # Calculate metrics
    results = {
        'overall_precision': precision_score(out_label_list, preds_list),
        'overall_recall': recall_score(out_label_list, preds_list),
        'overall_f1': f1_score(out_label_list, preds_list),
    }

    # Get detailed classification report
    report = classification_report(out_label_list, preds_list, output_dict=True)
    
    # Add entity-specific metrics
    for entity in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
        entity_metrics = {}
        for prefix in ['B-', 'I-']:
            label = prefix + entity
            if label in report:
                entity_metrics[prefix] = report[label]
        
        if entity_metrics:
            avg_f1 = np.mean([m['f1-score'] for m in entity_metrics.values()])
            avg_precision = np.mean([m['precision'] for m in entity_metrics.values()])
            avg_recall = np.mean([m['recall'] for m in entity_metrics.values()])
            
            results[f'{entity.lower()}_f1'] = avg_f1
            results[f'{entity.lower()}_precision'] = avg_precision
            results[f'{entity.lower()}_recall'] = avg_recall

    return results

def plot_confusion_matrix(true_labels: List[str], predicted_labels: List[str], output_path: str):
    """Plot confusion matrix for each entity type."""
    # Create confusion matrix
    labels = sorted(list(set(true_labels + predicted_labels)))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(trainer, test_dataset, output_dir: str):
    """Comprehensive model evaluation."""
    print("\nStarting model evaluation...")
    
    # Get predictions
    predictions, labels, _ = trainer.predict(test_dataset)
    preds_list, out_label_list = align_predictions(predictions, labels)
    
    # Flatten lists for overall metrics
    flat_preds = [item for sublist in preds_list for item in sublist]
    flat_labels = [item for sublist in out_label_list for item in sublist]
    
    # Calculate metrics
    metrics = compute_metrics((predictions, labels))
    
    # Create evaluation report
    report = {
        'Overall Metrics': {
            'Precision': metrics['overall_precision'],
            'Recall': metrics['overall_recall'],
            'F1 Score': metrics['overall_f1']
        },
        'Entity-wise Metrics': {}
    }
    
    # Add entity-specific metrics
    for entity in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
        if f'{entity.lower()}_f1' in metrics:
            report['Entity-wise Metrics'][entity] = {
                'F1 Score': metrics[f'{entity.lower()}_f1'],
                'Precision': metrics[f'{entity.lower()}_precision'],
                'Recall': metrics[f'{entity.lower()}_recall']
            }
    
    # Save metrics to file
    with open(f'{output_dir}/evaluation_report.txt', 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=====================\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("--------------\n")
        for metric, value in report['Overall Metrics'].items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nEntity-wise Metrics:\n")
        f.write("------------------\n")
        for entity, metrics in report['Entity-wise Metrics'].items():
            f.write(f"\n{entity}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        flat_labels, 
        flat_preds,
        f'{output_dir}/confusion_matrix.png'
    )
    
    # Create performance visualization
    entity_f1_scores = {
        entity: metrics[f'{entity.lower()}_f1']
        for entity in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']
        if f'{entity.lower()}_f1' in metrics
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(entity_f1_scores.keys(), entity_f1_scores.values())
    plt.title('F1 Score by Entity Type')
    plt.xlabel('Entity Type')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/entity_performance.png')
    plt.close()
    
    print(f"\nEvaluation complete! Reports saved to {output_dir}")
    return report


def compute_metrics(eval_pred) -> Dict:
    """Compute metrics for NER evaluation."""
    predictions, labels = eval_pred
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Align predictions and labels
    preds_list, out_label_list = align_predictions(predictions, labels)

    # Calculate metrics
    results = {
        'overall_precision': precision_score(out_label_list, preds_list),
        'overall_recall': recall_score(out_label_list, preds_list),
        'overall_f1': f1_score(out_label_list, preds_list),
    }

    # Get detailed classification report
    report = classification_report(out_label_list, preds_list, output_dict=True)
    
    # Add entity-specific metrics
    for entity in ['AGE', 'GENDER', 'ADDRESS', 'SKILLS', 'EXPERIENCE', 'EDUCATION', 'CERTIFICATION']:
        entity_metrics = {}
        for prefix in ['B-', 'I-']:
            label = prefix + entity
            if label in report:
                entity_metrics[prefix] = report[label]
        
        if entity_metrics:
            avg_f1 = np.mean([m['f1-score'] for m in entity_metrics.values()])
            avg_precision = np.mean([m['precision'] for m in entity_metrics.values()])
            avg_recall = np.mean([m['recall'] for m in entity_metrics.values()])
            
            results[f'{entity.lower()}_f1'] = avg_f1
            results[f'{entity.lower()}_precision'] = avg_precision
            results[f'{entity.lower()}_recall'] = avg_recall

    return results

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

def train_and_evaluate_roberta_ner(conll_file_path):
    """Train and evaluate RoBERTa NER model using custom CoNLL format data"""
    # Create output directories
    output_dir = "models/RoBERTa-fine-tuned-model"
    eval_dir = f"{output_dir}/evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load tokenizer and model
    print("Loading RoBERTa model and tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(
        'roberta-base',
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )
    
    # Load CoNLL data
    print("Loading custom CoNLL data...")
    df = read_custom_conll_file(conll_file_path)
    
    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_df)
    val_dataset = prepare_dataset(val_df)
    test_dataset = prepare_dataset(test_df)
    
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
    test_tokenized = test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        # compute_metrics=compute_metrics
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate model
    print("Evaluating model...")
    evaluation_results = evaluate_model(trainer, test_tokenized, eval_dir)
    
    # Print summary of results
    print("\nEvaluation Summary:")
    print(f"Overall F1 Score: {evaluation_results['Overall Metrics']['F1 Score']:.4f}")
    print("\nEntity-wise F1 Scores:")
    for entity, metrics in evaluation_results['Entity-wise Metrics'].items():
        print(f"{entity}: {metrics['F1 Score']:.4f}")
    
    print(f"\nDetailed evaluation reports and visualizations saved to {eval_dir}")
    print("Training and evaluation complete!")

if __name__ == "__main__":
    # Specify your CoNLL file path
    conll_file_path = "training/all-combined.conll"
    train_and_evaluate_roberta_ner(conll_file_path)
