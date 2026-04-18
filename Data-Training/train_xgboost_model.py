import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter

# Define paths
DATA_PATH = "./for-docu/with-jaccard-score-dataset.csv"
MODEL_OUTPUT_DIR = "./models/XGBoost-trained-model"
MODEL_FILE = os.path.join(MODEL_OUTPUT_DIR, "xgboost_job_matching_multi_class_model.json")

# Ensure model directory exists
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nSample data:")
print(df.head(2))
print("\nColumns:")
print(df.columns.tolist())

# Extract features based on the resume assessment pipeline
print("\nPreparing features...")

# Define feature columns based on entity scores
feature_columns = [
    'Age Score',           # binary: 1.0 if present
    'Gender Score',        # binary: 1.0 if present
    'Address Score',       # binary: 1.0 if present
    'Skills Score',        # Jaccard similarity with reference skills
    'Experience Score',    # Jaccard similarity with reference experience
    'Education Score',     # Jaccard similarity with reference education
    'Certification Score'  # Jaccard similarity with reference certifications
]

# Check if all feature columns exist in the dataset
missing_columns = [col for col in feature_columns if col not in df.columns]
if missing_columns:
    print(f"Warning: The following feature columns are missing: {missing_columns}")
    print("Available columns that might be similar:")
    for col in df.columns:
        if any(missing.lower() in col.lower() for missing in missing_columns):
            print(f"  - {col}")
    
    # Try to map column names if they exist with different capitalization
    for i, col in enumerate(feature_columns):
        for df_col in df.columns:
            if col.lower() == df_col.lower():
                feature_columns[i] = df_col
                break

# Create target variable based on suitability
print("\nCreating target variable for suitability classification...")
if 'Suitability' in df.columns:
    # Map suitability categories to numeric values
    suitability_map = {
        'Highly Suitable': 3,
        'Mildly Suitable': 2,
        'Less Suitable': 1,
        'Not Suitable': 0
    }
    
    # Check for unique values in Suitability column
    unique_suitability = df['Suitability'].unique()
    print(f"Unique suitability values in dataset: {unique_suitability}")
    
    # Map suitability to numeric target
    df['target'] = df['Suitability'].map(suitability_map)
    
    # Check if any NaN values in target
    if df['target'].isna().any():
        print(f"Warning: {df['target'].isna().sum()} NaN values in target. Filling with 0 (Not Suitable).")
        df['target'] = df['target'].fillna(0)
    
    # Check if 'Highly Suitable' class exists
    if 3 not in df['target'].unique():
        print("'Highly Suitable' class not found in dataset. Creating synthetic examples...")
        
        # Calculate total score for each row (sum of all feature scores)
        df['total_feature_score'] = df[feature_columns].sum(axis=1)
        
        # Find the top 50 'Mildly Suitable' candidates with highest scores
        mildly_suitable_mask = df['target'] == 2
        top_mildly_suitable = df[mildly_suitable_mask].nlargest(50, 'total_feature_score')
        
        # Upgrade them to 'Highly Suitable'
        df.loc[top_mildly_suitable.index, 'target'] = 3
        df.loc[top_mildly_suitable.index, 'Suitability'] = 'Highly Suitable'
        
        print(f"Created {len(top_mildly_suitable)} synthetic 'Highly Suitable' examples.")
else:
    print("Error: 'Suitability' column not found in dataset. This column is required.")
    exit(1)

# Display target distribution
print(f"Original suitability distribution: {df['target'].value_counts().to_dict()}")

# Prepare features and target
X = df[feature_columns].copy()
y = df['target'].astype(int)

# Fill any missing values in features with 0
X = X.fillna(0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape before sampling: {X_train.shape}")
print(f"Training set class distribution before sampling: {Counter(y_train)}")

# Apply SMOTE to handle class imbalance
print("\nApplying SMOTE oversampling to balance the dataset...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Training set shape after sampling: {X_train_resampled.shape}")
print(f"Training set class distribution after sampling: {Counter(y_train_resampled)}")
print(f"Test set shape: {X_test.shape}")
print(f"Test set class distribution: {Counter(y_test)}")

# Get the number of unique classes
num_classes = len(y.unique())
print(f"Number of unique classes: {num_classes}")

# Convert data to DMatrix format (required by XGBoost)
dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled, feature_names=feature_columns)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

# Define XGBoost parameters
params = {
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': num_classes,      # Use the actual number of classes
    'max_depth': 4,                # Maximum depth of a tree
    'eta': 0.1,                    # Learning rate
    'subsample': 0.8,              # Subsample ratio of the training instances
    'colsample_bytree': 0.8,       # Subsample ratio of columns when constructing each tree
    'min_child_weight': 1,         # Minimum sum of instance weight needed in a child
    'gamma': 0,                    # Minimum loss reduction required to make a further partition
    'seed': 42                     # Random seed
}

# Train the model
print("\nTraining XGBoost model...")
num_rounds = 100
model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=10
)

# Save the model
print(f"\nSaving model to {MODEL_FILE}...")
model.save_model(MODEL_FILE)

# Make predictions on test set
y_pred = model.predict(dtest)
y_pred = y_pred.astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy:.4f}")

# Define target names based on the suitability mapping
target_names = ['Not Suitable', 'Less Suitable', 'Mildly Suitable', 'Highly Suitable']
target_names = target_names[:num_classes]  # In case not all classes are present

# Print classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df.round(3))

# Save classification report as CSV
report_csv_path = os.path.join(MODEL_OUTPUT_DIR, 'classification_report.csv')
report_df.to_csv(report_csv_path)
print(f"Classification report saved to {report_csv_path}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
cm_path = os.path.join(MODEL_OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path)
print(f"Confusion matrix saved to {cm_path}")

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
cm_norm_path = os.path.join(MODEL_OUTPUT_DIR, 'normalized_confusion_matrix.png')
plt.savefig(cm_norm_path)
print(f"Normalized confusion matrix saved to {cm_norm_path}")

# Feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(model)
plt.title('Feature Importance')
plt.tight_layout()
fi_path = os.path.join(MODEL_OUTPUT_DIR, 'feature_importance.png')
plt.savefig(fi_path)
print(f"Feature importance plot saved to {fi_path}")

# Create a mapping from numeric classes to suitability levels
suitability_mapping = {v: k for k, v in suitability_map.items()}

# Save model metadata
model_metadata = {
    'feature_columns': feature_columns,
    'target_mapping': suitability_mapping,
    'model_params': params,
    'num_rounds': num_rounds,
    'accuracy': float(accuracy),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(os.path.join(MODEL_OUTPUT_DIR, 'model_metadata.json'), 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("\nModel training complete!")
print(f"Model and metadata saved to {MODEL_OUTPUT_DIR}")

# Create a prediction function for future use
def predict_suitability(features, model_path=MODEL_FILE):
    """
    Predict candidate suitability using the trained XGBoost model.
    
    Parameters:
    - features: dict with keys matching the feature columns
    - model_path: path to the saved model file
    
    Returns:
    - predicted_class: int (0-3)
    - suitability: string (Not Suitable, Less Suitable, etc.)
    - confidence: float (0-100%)
    """
    # Load the model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Prepare features
    feature_array = np.array([[
        features.get('Age Score', 0),
        features.get('Gender Score', 0),
        features.get('Address Score', 0),
        features.get('Skills Score', 0),
        features.get('Experience Score', 0),
        features.get('Education Score', 0),
        features.get('Certification Score', 0)
    ]])
    
    # Convert to DMatrix
    dfeatures = xgb.DMatrix(feature_array, feature_names=feature_columns)
    
    # Make prediction
    predicted_class = int(model.predict(dfeatures)[0])
    
    # Map class to suitability
    suitability = suitability_mapping.get(predicted_class, 'Unknown')
    
    # Get confidence scores for all classes
    pred_probs = model.predict(dfeatures, output_margin=True)
    pred_probs = np.exp(pred_probs) / np.sum(np.exp(pred_probs), axis=1, keepdims=True)
    confidence = float(pred_probs[0][predicted_class]) * 100
    
    # Apply score adjustments as per requirements
    has_cs_it_education = features.get('Education Score', 0) > 0
    has_relevant_experience = features.get('Experience Score', 0) > 0
    
    # Apply adjustments
    if not has_cs_it_education:
        confidence *= 0.5  # -50% if no CS/IT education
    
    if not has_relevant_experience:
        confidence *= 0.8  # -20% if no relevant experience
    
    return predicted_class, suitability, confidence

# Example usage of the prediction function
print("\nExample prediction:")
example_features = {
    'Age Score': 1.0,
    'Gender Score': 1.0,
    'Address Score': 1.0,
    'Skills Score': 0.8,
    'Experience Score': 0.7,
    'Education Score': 0.9,
    'Certification Score': 0.5
}
predicted_class, suitability, confidence = predict_suitability(example_features)
print(f"Predicted class: {predicted_class}")
print(f"Suitability: {suitability}")
print(f"Confidence: {confidence:.2f}%")

# Create a function to format output according to the frontend requirements
def format_assessment_output(suitability, confidence, entities):
    """
    Format the assessment output according to the frontend requirements.
    
    Parameters:
    - suitability: string (Not Suitable, Less Suitable, etc.)
    - confidence: float (0-100%)
    - entities: dict with extracted entity values
    
    Returns:
    - formatted_output: string
    """
    # Map suitability to job title according to requirements
    # Based on suitability level:
    # Highly Suitable or Mildly Suitable -> Programmer
    # Less Suitable or Not Suitable -> Encoder (with minimum 30% confidence)
    if suitability in ['Highly Suitable', 'Mildly Suitable'] or confidence >= 30:
        job_title = "Programmer"
    else:
        job_title = "Encoder"
        confidence = max(30, confidence)  # Minimum 30% for encoder role
    
    output = "->resume assessment:\n"
    output += f"  ->predicted suitable job: \"{job_title}\"\n"
    output += f"  ->prediction confidence: \"{int(confidence)}%\"\n"
    output += "  ->extracted information:\n"
    
    # Format entity values
    for entity, value in entities.items():
        if isinstance(value, list):
            value_str = ", ".join(value) if value else "none"
        else:
            value_str = value if value and value != "none" else "none"
        
        output += f"    ->{entity}: \"{value_str}\"\n"
    
    return output

# Example of formatted output
example_entities = {
    'age': '28',
    'gender': 'male',
    'address': 'Manila, Philippines',
    'skills': ['python', 'java', 'sql'],
    'education': 'computer science',
    'experience': 'software development',
    'certification': 'aws'
}

print("\nExample formatted output:")
print(format_assessment_output(suitability, confidence, example_entities))

# Plot feature importance as a bar chart
plt.figure(figsize=(12, 6))
importance_scores = model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': list(importance_scores.keys()),
    'Importance': list(importance_scores.values())
})
importance_df = importance_df.sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance (Gain)')
plt.tight_layout()
fi_bar_path = os.path.join(MODEL_OUTPUT_DIR, 'feature_importance_bar.png')
plt.savefig(fi_bar_path)
print(f"Feature importance bar chart saved to {fi_bar_path}")
