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
