import os
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from utilities.text_extraction import extract_text_from_file  # You need a text extraction function
from werkzeug.utils import secure_filename

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model("C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/XGBoost-model-training/trained-XGBoost-model/xgboost_model.json")  # Make sure this file exists

# Define categorical columns
CATEGORICAL_COLUMNS = ["Gender", "Address", "Skills", "Education", "Work Experience", "Certificates", "Course"]

def preprocess_resume(resume_path):
    """Extract and preprocess resume text."""
    resume_text = extract_text_from_file(resume_path)

    # Convert resume text into TF-IDF features
    vectorizer = TfidfVectorizer(max_features=8)  # Use same settings as training
    tfidf_features = vectorizer.fit_transform([resume_text]).toarray()

    # Create a DataFrame for extracted features
    resume_features = pd.DataFrame(tfidf_features, columns=[f"job_feature_{i}" for i in range(tfidf_features.shape[1])])

    return resume_features

def analyze_resumes(upload_folder="uploads"):
    """Analyze uploaded resumes using XGBoost."""
    results = []
    
    # Process all resumes in the upload folder
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)

        # Extract and preprocess resume text
        resume_features = preprocess_resume(file_path)

        # Load sample categorical data (Replace this with real user input)
        sample_data = {
            "Age": 30, "Gender": "Male", "Address": "CityX", "Skills": "Python, ML",
            "Education": "Bachelor's", "Work Experience": "5 years", "Certificates": "AWS",
            "Course": "Computer Science"
        }

        # Convert categorical values to numerical (using Label Encoding)
        for col in CATEGORICAL_COLUMNS:
            sample_data[col] = LabelEncoder().fit_transform([sample_data[col]])[0]

        # Convert to DataFrame and merge with resume features
        sample_df = pd.DataFrame([sample_data])
        final_features = pd.concat([sample_df, resume_features], axis=1)

        # Predict suitability score
        prediction = model.predict(final_features)[0]
        probability = model.predict_proba(final_features)[0][1]  # Probability of getting hired

        results.append({
            "filename": filename,
            "prediction": "Hired" if prediction == 1 else "Not Hired",
            "suitability_score": round(probability * 100, 2)
        })

    return results
