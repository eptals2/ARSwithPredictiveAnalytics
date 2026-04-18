import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# 🔹 Load the Excel file instead of CSV
df = pd.read_excel("C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Data-Training/dataset-inputs/csv-datasets/resume_from_PECIT.xlsx")

print("Initial Data Shape:", df.shape)

# 🔹 Clean missing values
# df.dropna(subset=["resume_text", "suitability_labels"], inplace=True)

# 🔹 TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["resume_text"])

# 🔹 Numeric labels (0 = not suitable, 1 = suitable)
y = df["suitability_label"]

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# After splitting
print("Train Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)
# 🔹 Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 🔹 Predict
y_pred = model.predict(X_test)
# After predictions
print("Predictions:", len(y_pred))
# 🔹 Show performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Suitable", "Suitable"]))

# 🔹 Save model and vectorizer
joblib.dump(model, "xgboost_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# 🔹 Optional: Save predictions to Excel
results = pd.DataFrame({
    "resume_text": df.iloc[y_test.index]["resume_text"].values,
    "actual_label": y_test.values,
    "predicted_label": y_pred
})

results.to_excel("resume_predictions.xlsx", index=False)

print("\n✅ Model trained and predictions saved to 'resume_predictions.xlsx'")
import joblib

# Load the model and vectorizer
model = joblib.load("C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Flask-App-Implementation/models/XGBoost-model/xgboost_model.pkl")
vectorizer = joblib.load("C:/Users/Acer/Desktop/ARSwithPredictiveAnalytics/Flask-App-Implementation/models/XGBoost-model/tfidf_vectorizer.pkl")

# New resume sample (as a list of 1 string)
new_resume = ["""PART-TIME LECTURER



PROFILE



OBJECTIVES: To secure employment

with a reputable company, where |!

can utilize my skills and develop my

knowledge as well as my

interpersonal relationship to my

colleagues.



CONTACT

bat airnleni ae nae ee

PHONE:



EMAIL:



EDUCATION



2017 - PRES!

MASTER OF SCIENCE IN INFORMATION TECHNOLOGY-CAR



CARAGA STATE UNIVERSITY



2014- 2015



CERTIFICATE OF TEACHING EDUCATION

(33 COMPLETED UNITS)



CARAGA STATE UNIVERSITY

2003-2011

BACHELOR OF SCIENCE IN COMPUTER SCIENCE



WORK EXPERIENCE



DEPARTMENT OF EDUCATION

SENIOR HIGH TEACHER

2016-PRESENT



Teaching .Net Programming, Computer System Servicing, Electronics

and Robotics and Mathematics Subject



Senior High Computer Laboratory in-charge



HOLY CHILD COLLEGES OF BUTUAN

PART-TIME LECTURER



2023-2024



Teaching System Integration and Architecture



CARAGA STATE UNIVERSITY-MAIN- CCIS



PART-TIME LECTURER



2016-2023



Teaching Fundamental Programming, Discrete Mathematics



CARAGA STATE UNIVERSITY-MAIN

FACILITATOR



2014-2023



Teaching NSTP



CARAGA STATE UNIVERSITY-MAIN- VP ADMIN OFFICE

TECHNICAL ASSISTANT



2010-2016



Technical Staff Work



TMC HOLDER - COMPUTER SYSTEM SERVICING

NC 2 HOLDER - COMPUTER SYSTEM SERVICING



NC3 -HOLDER- VISUAL GRAPHICS DESIGN
"""]

# Transform and predict
X_new = vectorizer.transform(new_resume)
y_pred = model.predict(X_new)

# Output the result
label = "Suitable" if y_pred[0] == 1 else "Not Suitable"
print("Prediction:", label)
