# Resume Analysis System with Predictive Analytics

A Flask-based web application that extracts information from resumes using RoBERTa NER (Named Entity Recognition) and performs candidate suitability assessment using XGBoost predictive analytics.

## Features

- **Resume Information Extraction**: Extract key information from resumes using RoBERTa NER including:
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

- **Multiple Resume Analysis**: Upload and analyze multiple resumes at once to:
  - Compare candidates side by side
  - Identify the best candidate based on overall score
  - View detailed assessment for each candidate

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

3. Upload one or multiple resume files (PDF, DOCX, or TXT format).
   - You can select multiple files by holding Ctrl (or Cmd on Mac) while selecting files.

4. Click the "Extract & Analyze Resumes" button to process all the uploaded resumes.

5. Review the list of candidates with their overall scores and suitability ratings.

6. Click "View Details" on any candidate card to see the detailed assessment, including extracted information, category scores, strengths, areas for improvement, and recommendation.

7. If you've uploaded multiple resumes, click the "Compare Candidates" button to see a side-by-side comparison of all candidates.

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
- Add more advanced comparison metrics and visualizations
- Implement filtering and sorting options for candidate comparison
- Add export functionality for assessment reports
