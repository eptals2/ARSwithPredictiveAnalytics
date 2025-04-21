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
