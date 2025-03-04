# Resume Analysis System with Predictive Analytics

A Flask-based web application that extracts information from resumes using RoBERTa NER (Named Entity Recognition) and performs candidate suitability assessment using XGBoost predictive analytics.

## Features

- **Resume Information Extraction**: Extract key information from resumes including:
  - Age
  - Gender
  - Address
  - Soft Skills
  - Hard Skills
  - Education Level
  - Course/Major
  - Experience
  - Certifications

- **Candidate Assessment**: Analyze extracted information to assess candidate suitability using XGBoost predictive analytics, providing:
  - Overall suitability score
  - Category-wise scores
  - Strengths and areas for improvement
  - Hiring recommendation

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd ARSwithPredictiveAnalytics
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Upload a resume file (PDF, DOCX, or TXT format).

4. Click the "Extract Information" button to extract key information from the resume.

5. Review the extracted information and click "Analyze Candidate" to perform the suitability assessment.

6. View the assessment results, including the overall score, category scores, strengths, areas for improvement, and recommendation.

## Technical Details

### Technologies Used

- **Backend**: Flask (Python)
- **NER Model**: RoBERTa (Simulated for demonstration)
- **Predictive Analytics**: XGBoost (Simulated for demonstration)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Document Processing**: PyPDF2 (PDF), python-docx (DOCX)

### Project Structure

- `app.py`: Main Flask application file
- `templates/index.html`: HTML template for the web interface
- `uploads/`: Directory for storing uploaded resume files
- `requirements.txt`: List of Python dependencies

## Notes

- This application uses simulated models for both NER and XGBoost for demonstration purposes.
- In a production environment, you would need to train and fine-tune actual models for better accuracy.
- The application supports PDF, DOCX, and TXT file formats for resume uploads.

## Future Improvements

- Implement and fine-tune an actual RoBERTa model for NER
- Train a real XGBoost model with labeled data for more accurate predictions
- Add support for more file formats
- Implement user authentication and resume storage
- Add ability to compare multiple candidates
- Provide more detailed analysis and visualizations
