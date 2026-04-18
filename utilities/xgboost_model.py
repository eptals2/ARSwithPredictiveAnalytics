"""Module for job suitability prediction using Flask and machine learning."""

import xgboost as xgb
import pandas as pd
import os
from utilities.pre_processing import extract_text_from_file, preprocess_text
from utilities.similarity_utils import jaccard_similarity  # Import from the new file

MODEL_PATH = "xgboost_ranking_model.json"

xgb_model = xgb.Booster()
if os.path.exists(MODEL_PATH):
    xgb_model.load_model(MODEL_PATH)
else:
    print(f"Warning: XGBoost model not found at {MODEL_PATH}")

def predict_xgboost_score(job_path, resume_path):
    """Predict ranking score using XGBoost model"""
    job_text = extract_text_from_file(job_path)
    resume_text = extract_text_from_file(resume_path)

    job_features = preprocess_text(job_text)
    resume_features = preprocess_text(resume_text)

    feature_data = pd.DataFrame([{
        "jaccard_score": jaccard_similarity(set(job_features), set(resume_features)),
    }])

    expected_columns = xgb_model.feature_names
    feature_data = feature_data.reindex(columns=expected_columns, fill_value=0)

    dmatrix = xgb.DMatrix(feature_data)
    return xgb_model.predict(dmatrix)[0]
