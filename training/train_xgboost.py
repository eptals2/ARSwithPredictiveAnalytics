import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# Reference entities from memory
REFERENCE_ENTITIES = {
    'skills': ['python', 'java', 'javascript', 'c++', 'sql', 'html', 'css', 'web development'],
    'experience': ['software development', 'programming', 'web development', 'internship'],
    'education': ['computer science', 'information technology', 'software engineering'],
    'certifications': ['microsoft', 'aws', 'oracle', 'cisco', 'comptia']
}

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def preprocess_features(df):
    """Convert raw resume data into feature vectors"""
    features = []
    
    for _, row in df.iterrows():
        # Convert strings to sets for comparison
        candidate_skills = set(str(row['skills']).lower().split(','))
        candidate_exp = set(str(row['experience']).lower().split(','))
        candidate_edu = set(str(row['education']).lower().split(','))
        candidate_cert = set(str(row['certifications']).lower().split(','))
        
        # Calculate similarities
        feature_vector = {
            'age_similarity': 1.0 if pd.notna(row['age']) else 0.0,
            'gender_similarity': 1.0 if pd.notna(row['gender']) else 0.0,
            'address_similarity': 1.0 if pd.notna(row['address']) else 0.0,
            'skills_similarity': calculate_jaccard_similarity(
                candidate_skills, set(REFERENCE_ENTITIES['skills'])),
            'experience_similarity': calculate_jaccard_similarity(
                candidate_exp, set(REFERENCE_ENTITIES['experience'])),
            'education_similarity': calculate_jaccard_similarity(
                candidate_edu, set(REFERENCE_ENTITIES['education'])),
            'certification_similarity': calculate_jaccard_similarity(
                candidate_cert, set(REFERENCE_ENTITIES['certifications']))
        }
        features.append(feature_vector)
    
    return pd.DataFrame(features)

def apply_score_adjustments(predictions, education_data, experience_data):
    """Apply score adjustments based on rules"""
    adjusted_scores = []
    
    for pred, edu, exp in zip(predictions, education_data, experience_data):
        score = pred * 100  # Convert to percentage
        
        # Education penalty
        edu_lower = str(edu).lower()
        if not any(term in edu_lower for term in ['computer science', 'information technology']):
            score *= 0.5  # -50% penalty
            
        # Experience penalty
        exp_lower = str(exp).lower()
        if exp_lower == 'none' or pd.isna(exp):
            score *= 0.8  # -20% penalty
            
        # Role selection based on score
        if score >= 30:
            role = 'Programmer'
        else:
            role = 'Encoder'
            score = max(30, score)  # Minimum 30% for encoder
            
        adjusted_scores.append({'role': role, 'confidence': score})
    
    return adjusted_scores

def train_model():
    # Load training data
    df = pd.read_csv('training_data.csv')
    
    # Preprocess features
    X = preprocess_features(df)
    y = (df['job_role'] == 'Programmer').astype(int)  # Binary classification
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    
    # Apply adjustments
    train_adjusted = apply_score_adjustments(
        train_preds, 
        df.iloc[X_train.index]['education'],
        df.iloc[X_train.index]['experience']
    )
    
    test_adjusted = apply_score_adjustments(
        test_preds,
        df.iloc[X_test.index]['education'],
        df.iloc[X_test.index]['experience']
    )
    
    # Save model
    model.save_model('models/XGBoost-trained-model/xgboost_job_matching_multi_class_model.json')
    
    # Print evaluation metrics
    print("Training Accuracy:", accuracy_score(
        y_train, 
        [1 if adj['role'] == 'Programmer' else 0 for adj in train_adjusted]
    ))
    print("Testing Accuracy:", accuracy_score(
        y_test,
        [1 if adj['role'] == 'Programmer' else 0 for adj in test_adjusted]
    ))

if __name__ == "__main__":
    train_model()
