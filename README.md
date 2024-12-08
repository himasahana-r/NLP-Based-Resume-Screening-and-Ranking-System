# Resume Matcher with Explainable AI

A **Streamlit** application that matches resumes to job descriptions using **SBERT** embeddings and **Bag-of-Words (BoW)** similarity techniques. The app integrates **Explainable AI (XAI)** features to provide insights into the matching process, including matched and missing skills for resumes.

---

## Features

1. **Resume Upload**:
   - Upload multiple resumes in PDF format.
   - Extracts text and preprocesses resumes.
2. **Job Description Integration**:
   - Preload job descriptions from a dataset of job titles and descriptions.
   - Option to input a custom job description.
3. **Skills Extraction**:
   - Extracts relevant skills from job descriptions and resumes using predefined skill dictionaries.
4. **Similarity Matching**:
   - **SBERT (Sentence-BERT)** embeddings for semantic similarity.
   - **Bag-of-Words (BoW)** for fallback similarity computation.
5. **Explainable AI (XAI)**:
   - Displays similarity scores for all resumes.
   - Highlights the top-matched resume.
   - Explains matched and missing skills for the top match.
6. **Evaluation Metrics**:
   - Evaluate performance with **Precision**, **Recall**, and **F1 Score** against a gold standard dataset.

---

## Installation

### Prerequisites
- Python 3.9 or later
- Required Python libraries (see `requirements.txt`)

### Steps

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/resume-matcher.git
    ```

2. Navigate to the project directory:
    ```bash
    cd resume-matcher
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    streamlit run resume_matcher.py
    ```
5. Download the .gz files, extract them and place the dataset files in the main directory as the resume_matcher.py.

6. Replace the path in the code with the path of the downloaded dataset file.
---

## Usage

### 1. Upload Resumes
- Upload multiple PDF resumes for analysis.

### 2. Select or Enter Job Description
- Choose a job title from the preloaded dataset or manually enter a job description.

### 3. Match Resumes
- Click the **Match Resumes** button to compute similarity scores and display results.

### 4. View Results
- **Similarity Scores**: View scores for all uploaded resumes.
- **Explainability**: See matched and missing skills for the top-matched resume.

### 5. Evaluation Metrics
- Upload a gold standard dataset for performance evaluation.
- Adjust the relevance threshold dynamically and view Precision, Recall, and F1 Score.
