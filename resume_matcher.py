import streamlit as st
import os
import re
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy

# Load SpaCy and SBERT models
nlp = spacy.load("en_core_web_sm")
try:
    sbert_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2')
    sbert_available = True
except Exception as e:
    st.warning(f"SBERT Model not loaded. Using Bag-of-Words as fallback. Error: {e}")
    sbert_available = False

# Skills Dictionary
skill_lemma_dict = {
    # Programming Languages
    "python": ["python", "py", "pandas", "numpy", "scipy", "flask", "django"],
    "java": ["java", "jvm", "spring", "hibernate", "maven", "gradle"],
    "javascript": ["javascript", "js", "node.js", "react.js", "vue.js", "angular", "typescript"],
    "csharp": ["c#", ".net", "asp.net", "entity framework"],
    "cpp": ["c++", "cpp", "qt", "boost"],
    "php": ["php", "laravel", "symfony"],
    "ruby": ["ruby", "rails"],
    "go": ["go", "golang"],
    "swift": ["swift", "ios"],
    "kotlin": ["kotlin", "android"],
    "scala": ["scala", "akka", "play framework"],
    "rust": ["rust"],
    "perl": ["perl"],

    # Frameworks and Libraries
    "react": ["react", "react.js", "react native"],
    "angular": ["angular"],
    "vue": ["vue", "vue.js"],
    "jquery": ["jquery"],
    "bootstrap": ["bootstrap"],
    "express": ["express"],
    "spring boot": ["spring boot"],
    "tensorflow": ["tensorflow"],
    "keras": ["keras"],
    "pytorch": ["pytorch"],
    "scikit-learn": ["scikit-learn"],

    # Databases
    "sql": ["sql", "mysql", "postgresql", "oracle", "sql server"],
    "nosql": ["nosql", "mongodb", "cassandra", "redis", "neo4j", "couchdb"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql"],
    "mongodb": ["mongodb"],
    "redis": ["redis"],
    "sqlite": ["sqlite"],
    "oracle": ["oracle"],
    "snowflake": ["snowflake"],
    "redshift": ["redshift"],
    "greenplum": ["greenplum"],
    "teradata": ["teradata"],

    # DevOps and Cloud Platforms
    "aws": ["aws", "ec2", "s3", "lambda", "rds"],
    "azure": ["azure", "azure devops"],
    "google cloud": ["google cloud", "gcp", "app engine", "kubernetes engine"],
    "docker": ["docker", "docker-compose"],
    "kubernetes": ["kubernetes", "k8s"],
    "jenkins": ["jenkins"],
    "travis ci": ["travis ci"],
    "gitlab ci": ["gitlab ci"],
    "circleci": ["circleci"],

    # Web Technologies
    "html": ["html"],
    "css": ["css", "sass", "less"],
    "rest api": ["rest", "restful", "json", "xml"],
    "graphql": ["graphql"],

    # Software Development Methodologies
    "agile": ["agile", "scrum", "kanban"],
    "devops": ["devops", "site reliability"],

    # Soft Skills
    "leadership": ["leadership", "management"],
    "communication": ["communication", "teamwork"],
    "problem-solving": ["problem-solving", "analytical skills"],
    "adaptability": ["adaptability", "flexibility"],
    "teamwork": ["teamwork"],
    "leadership": ["leadership"],
    "project management": ["project management", "pm"],
    "creativity": ["creativity"],
    "critical thinking": ["critical thinking"],
    "emotional intelligence": ["emotional intelligence", "eq"],
    "negotiation": ["negotiation"],
    "decision making": ["decision making"],
    
    # Machine Learning
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "reinforcement learning": ["reinforcement learning", "rl"],
    "supervised learning": ["supervised learning"],
    "unsupervised learning": ["unsupervised learning"],
    "semi-supervised learning": ["semi-supervised learning"],
    "natural language processing": ["natural language processing", "nlp"],
    "computer vision": ["computer vision"],
    "speech recognition": ["speech recognition"],
    "anomaly detection": ["anomaly detection"],
    "generative adversarial networks": ["gan", "generative adversarial networks"],
    "transfer learning": ["transfer learning"],
    "feature engineering": ["feature engineering"],
    "model optimization": ["model optimization"],
    "model deployment": ["model deployment"],
    "edge AI": ["edge ai", "edge computing"],
    "federated learning": ["federated learning"],
    "explainable AI": ["explainable ai", "xai"],

    # ML Frameworks and Libraries
    "tensorflow": ["tensorflow", "tf"],
    "keras": ["keras"],
    "pytorch": ["pytorch"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scipy": ["scipy"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "opencv": ["opencv"],
    "spacy": ["spacy"],
    "nltk": ["nltk"],
    "gensim": ["gensim"],
    "huggingface": ["huggingface", "transformers"],
    "fastai": ["fastai"],
    "caffe": ["caffe"],
    "theano": ["theano"],
    "dlib": ["dlib"],
    "mlflow": ["mlflow"],
    "pycaret": ["pycaret"],
    "streamlit": ["streamlit"],
    "dash": ["dash"],
    
    # Time Series Analysis & Forecasting
    "time series": ["time series", "time series analysis"],
    "arima": ["arima"],
    "prophet": ["prophet"],
    "lstm": ["lstm"],
    
    # Deep Learning Specific Technologies
    "convolutional neural networks": ["cnn", "convolutional neural networks"],
    "recurrent neural networks": ["rnn", "recurrent neural networks"],
    "long short-term memory": ["lstm"],
    "transformers": ["transformers"],
    "bert": ["bert"],
    "gpt": ["gpt", "gpt-2", "gpt-3"],

    # Visualization and Reporting Tools
    "tableau": ["tableau"],
    "power bi": ["power bi"],
    "qlik": ["qlik", "qlikview", "qliksense"],
    "looker": ["looker"],
    
    # Big Data
    "big data": ["big data"],
    "hadoop": ["hadoop"],
    "spark": ["spark"],
    "kafka": ["kafka"],
    "hive": ["hive"],
    "flink": ["flink"],
    "elastic search": ["elastic search", "elasticsearch"],
    "solr": ["solr"],
    "cassandra": ["cassandra"],
    "hbase": ["hbase"],
    "neo4j": ["neo4j"],

    # Cloud and DevOps
    "aws": ["aws", "amazon web services"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud platform"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "jenkins": ["jenkins"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous deployment"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "cloudformation": ["cloudformation"],
    "openstack": ["openstack"],

    # Data Engineering and ETL Tools
    "airflow": ["airflow"],
    "luigi": ["luigi"],
    "talend": ["talend"],
    "pentaho": ["pentaho"],
    "informatica": ["informatica"],

    # Miscellaneous
    "git": ["git", "version control", "svn"],
    "project management": ["jira", "trello", "asana"],
    "ui/ux": ["ui", "ux", "user interface", "user experience"],
    "cybersecurity": ["cybersecurity", "penetration testing", "encryption"],
    "data analysis": ["data analysis", "excel", "power bi", "tableau", "data analytics"],
    "data visualization": ["data visualization"],
    "data mining": ["data mining"],
    "statistical analysis": ["statistical analysis", "statistics"],
    "ethics in AI": ["ethics in ai", "ai ethics"],
    "quantum computing": ["quantum computing"],
    "blockchain": ["blockchain"],
    "augmented reality": ["augmented reality", "ar"],
    "virtual reality": ["virtual reality", "vr"],
    "internet of things": ["iot", "internet of things"],
    "robotics": ["robotics"],
    "drones": ["drones", "uav"],
    "penetration testing": ["penetration testing"],
    "blockchain": ["blockchain"],
    "cryptocurrency": ["cryptocurrency", "bitcoin", "ethereum"],
    "quantum computing": ["quantum computing"],
    "bioinformatics": ["bioinformatics"],
    "digital twin": ["digital twin"],
    "autonomous vehicles": ["autonomous vehicles", "self-driving cars"]
}

# Regex Skills Dictionary
skill_regex_dict = {
    "python": r"\b(python|py|pandas|numpy|scipy|flask|django)\b",
    "java": r"\b(java|jvm|spring|hibernate|maven|gradle)\b",
    "javascript": r"\b(javascript|js|node\.js|react\.js|vue\.js|angular|typescript)\b",
    "csharp": r"\b(c#|\.net|asp\.net|entity framework)\b",
    "cpp": r"\b(c\+\+|cpp|qt|boost)\b",
    "php": r"\b(php|laravel|symfony)\b",
    "ruby": r"\b(ruby|rails)\b",
    "go": r"\b(go|golang)\b",
    "swift": r"\b(swift|ios)\b",
    "kotlin": r"\b(kotlin|android)\b",
    "scala": r"\b(scala|akka|play framework)\b",
    "rust": r"\b(rust)\b",
    "perl": r"\b(perl)\b",
    "react": r"\b(react|react\.js|react native)\b",
    "angular": r"\b(angular)\b",
    "vue": r"\b(vue|vue\.js)\b",
    "jquery": r"\b(jquery)\b",
    "bootstrap": r"\b(bootstrap)\b",
    "express": r"\b(express)\b",
    "spring boot": r"\b(spring boot)\b",
    "tensorflow": r"\b(tensorflow)\b",
    "keras": r"\b(keras)\b",
    "pytorch": r"\b(pytorch)\b",
    "scikit-learn": r"\b(scikit-learn)\b",
    "sql": r"\b(sql|mysql|postgresql|oracle|sql server)\b",
    "nosql": r"\b(nosql|mongodb|cassandra|redis|neo4j|couchdb)\b",
    "mysql": r"\b(mysql)\b",
    "postgresql": r"\b(postgresql)\b",
    "mongodb": r"\b(mongodb)\b",
    "redis": r"\b(redis)\b",
    "sqlite": r"\b(sqlite)\b",
    "oracle": r"\b(oracle)\b",
    "snowflake": r"\b(snowflake)\b",
    "redshift": r"\b(redshift)\b",
    "greenplum": r"\b(greenplum)\b",
    "teradata": r"\b(teradata)\b",
    "aws": r"\b(aws|ec2|s3|lambda|rds)\b",
    "azure": r"\b(azure|azure devops)\b",
    "google cloud": r"\b(google cloud|gcp|app engine|kubernetes engine)\b",
    "docker": r"\b(docker|docker-compose)\b",
    "kubernetes": r"\b(kubernetes|k8s)\b",
    "jenkins": r"\b(jenkins)\b",
    "travis ci": r"\b(travis ci)\b",
    "gitlab ci": r"\b(gitlab ci)\b",
    "circleci": r"\b(circleci)\b",
    "html": r"\b(html)\b",
    "css": r"\b(css|sass|less)\b",
    "rest api": r"\b(rest|restful|json|xml)\b",
    "graphql": r"\b(graphql)\b",
    "agile": r"\b(agile|scrum|kanban)\b",
    "devops": r"\b(devops|site reliability)\b",
    "leadership": r"\b(leadership|management)\b",
    "communication": r"\b(communication|teamwork)\b",
    "problem-solving": r"\b(problem-solving|analytical skills)\b",
    "adaptability": r"\b(adaptability|flexibility)\b",
    "teamwork": r"\b(teamwork)\b",
    "project management": r"\b(project management|pm)\b",
    "creativity": r"\b(creativity)\b",
    "critical thinking": r"\b(critical thinking)\b",
    "emotional intelligence": r"\b(emotional intelligence|eq)\b",
    "negotiation": r"\b(negotiation)\b",
    "decision making": r"\b(decision making)\b",
    "machine learning": r"\b(machine learning|ml)\b",
    "deep learning": r"\b(deep learning|dl)\b",
    "reinforcement learning": r"\b(reinforcement learning|rl)\b",
    "supervised learning": r"\b(supervised learning)\b",
    "unsupervised learning": r"\b(unsupervised learning)\b",
    "semi-supervised learning": r"\b(semi-supervised learning)\b",
    "natural language processing": r"\b(natural language processing|nlp)\b",
    "computer vision": r"\b(computer vision)\b",
    "speech recognition": r"\b(speech recognition)\b",
    "anomaly detection": r"\b(anomaly detection)\b",
    "generative adversarial networks": r"\b(gan|generative adversarial networks)\b",
    "transfer learning": r"\b(transfer learning)\b",
    "feature engineering": r"\b(feature engineering)\b",
    "model optimization": r"\b(model optimization)\b",
    "model deployment": r"\b(model deployment)\b",
    "edge AI": r"\b(edge ai|edge computing)\b",
    "federated learning": r"\b(federated learning)\b",
    "explainable AI": r"\b(explainable ai|xai)\b",
    "tensorflow": r"\b(tensorflow|tf)\b",
    "keras": r"\b(keras)\b",
    "pytorch": r"\b(pytorch)\b",
    "scikit-learn": r"\b(scikit-learn|sklearn)\b",
    "pandas": r"\b(pandas)\b",
    "numpy": r"\b(numpy)\b",
    "scipy": r"\b(scipy)\b",
    "matplotlib": r"\b(matplotlib)\b",
    "seaborn": r"\b(seaborn)\b",
    "xgboost": r"\b(xgboost)\b",
    "lightgbm": r"\b(lightgbm)\b",
    "opencv": r"\b(opencv)\b",
    "spacy": r"\b(spacy)\b",
    "nltk": r"\b(nltk)\b",
    "gensim": r"\b(gensim)\b",
    "huggingface": r"\b(huggingface|transformers)\b",
    "fastai": r"\b(fastai)\b",
    "caffe": r"\b(caffe)\b",
    "theano": r"\b(theano)\b",
    "dlib": r"\b(dlib)\b",
    "mlflow": r"\b(mlflow)\b",
    "pycaret": r"\b(pycaret)\b",
    "streamlit": r"\b(streamlit)\b",
    "dash": r"\b(dash)\b",
    "time series": r"\b(time series|time series analysis)\b",
    "arima": r"\b(arima)\b",
    "prophet": r"\b(prophet)\b",
    "lstm": r"\b(lstm)\b",
    "convolutional neural networks": r"\b(cnn|convolutional neural networks)\b",
    "recurrent neural networks": r"\b(rnn|recurrent neural networks)\b",
    "long short-term memory": r"\b(lstm)\b",
    "transformers": r"\b(transformers)\b",
    "bert": r"\b(bert)\b",
    "gpt": r"\b(gpt|gpt-2|gpt-3)\b",
    "tableau": r"\b(tableau)\b",
    "power bi": r"\b(power bi)\b",
    "qlik": r"\b(qlik|qlikview|qliksense)\b",
    "looker": r"\b(looker)\b",
    "big data": r"\b(big data)\b",
    "hadoop": r"\b(hadoop)\b",
    "spark": r"\b(spark)\b",
    "kafka": r"\b(kafka)\b",
    "hive": r"\b(hive)\b",
    "flink": r"\b(flink)\b",
    "elastic search": r"\b(elastic search|elasticsearch)\b",
    "solr": r"\b(solr)\b",
    "cassandra": r"\b(cassandra)\b",
    "hbase": r"\b(hbase)\b",
    "neo4j": r"\b(neo4j)\b",
    "aws": r"\b(aws|amazon web services)\b",
    "azure": r"\b(azure)\b",
        "gcp": r"\b(gcp|google cloud platform)\b",
    "docker": r"\b(docker)\b",
    "kubernetes": r"\b(kubernetes|k8s)\b",
    "jenkins": r"\b(jenkins)\b",
    "travis ci": r"\b(travis ci)\b",
    "gitlab ci": r"\b(gitlab ci)\b",
    "circleci": r"\b(circleci)\b",
    "ci/cd": r"\b(ci/cd|continuous integration|continuous deployment)\b",
    "terraform": r"\b(terraform)\b",
    "ansible": r"\b(ansible)\b",
    "cloudformation": r"\b(cloudformation)\b",
    "openstack": r"\b(openstack)\b",
    "airflow": r"\b(airflow)\b",
    "luigi": r"\b(luigi)\b",
    "talend": r"\b(talend)\b",
    "pentaho": r"\b(pentaho)\b",
    "informatica": r"\b(informatica)\b",
    "git": r"\b(git|version control|svn)\b",
    "project management": r"\b(project management|pm)\b",
    "ui/ux": r"\b(ui|ux|user interface|user experience)\b",
    "cybersecurity": r"\b(cybersecurity|penetration testing|encryption)\b",
    "data analysis": r"\b(data analysis|excel|power bi|tableau|data analytics)\b",
    "data visualization": r"\b(data visualization)\b",
    "data mining": r"\b(data mining)\b",
    "statistical analysis": r"\b(statistical analysis|statistics)\b",
    "ethics in AI": r"\b(ethics in ai|ai ethics)\b",
    "quantum computing": r"\b(quantum computing)\b",
    "blockchain": r"\b(blockchain)\b",
    "augmented reality": r"\b(augmented reality|ar)\b",
    "virtual reality": r"\b(virtual reality|vr)\b",
    "internet of things": r"\b(iot|internet of things)\b",
    "robotics": r"\b(robotics)\b",
    "drones": r"\b(drones|uav)\b",
    "penetration testing": r"\b(penetration testing)\b",
    "cryptocurrency": r"\b(cryptocurrency|bitcoin|ethereum)\b",
    "bioinformatics": r"\b(bioinformatics)\b",
    "digital twin": r"\b(digital twin)\b",
    "autonomous vehicles": r"\b(autonomous vehicles|self-driving cars)\b"
}   

# Preprocessing Function
def preprocess(text):
    text = re.sub(r'[^\x00-\x7f]', r' ', text).lower()
    text = re.sub(r'http\S+|@\S+|#\S+', ' ', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract Text from PDF
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Extract Skills Using the Skills Dictionary
def extract_skills(text):
    doc = nlp(text.lower())
    extracted_skills = set()
    for token in doc:
        for skill, regex in skill_regex_dict.items():
            if re.search(regex, token.text):
                extracted_skills.add(skill)
    return extracted_skills


# Calculate Similarity Using SBERT
def calculate_similarity_sbert(job_description, resumes):
    """
    Calculate similarity scores between job description and resumes using SBERT.
    Args:
        job_description (str): The job description.
        resumes (list of str): List of resumes' text.
    Returns:
        list of float: Cosine similarity scores for each resume.
    """
    job_embedding = sbert_model.encode([job_description])  # 2D array (1, embedding_size)
    resume_embeddings = sbert_model.encode(resumes)  # 2D array (n, embedding_size)

    # Ensure both arrays are 2D
    if job_embedding.ndim == 1:
        job_embedding = job_embedding.reshape(1, -1)  # Reshape to (1, embedding_size)
    if resume_embeddings.ndim == 1:
        resume_embeddings = resume_embeddings.reshape(1, -1)  # Reshape to (n, embedding_size)

    # Calculate cosine similarities
    scores = cosine_similarity(job_embedding, resume_embeddings)[0]  # Result is 1D array of scores
    return scores



# Convert Text to Bag-of-Words Vectors
def text_to_bow_vector(texts):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

# Match Resumes Using BoW
def match_resumes_with_job_bow(job_description, resume_files):
    job_desc_text = preprocess(job_description)
    resume_texts = [preprocess(extract_text_from_pdf(file)) for file in resume_files]

    all_texts = [job_desc_text] + resume_texts
    vectors, vectorizer = text_to_bow_vector(all_texts)

    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    similarities = [cosine_similarity(job_vector, vector)[0][0] for vector in resume_vectors]
    matched_resumes = sorted(zip([file.name for file in resume_files], similarities), key=lambda x: x[1], reverse=True)
    return matched_resumes

# Load the provided dataset
@st.cache_data
def load_job_dataset(dataset_path):
    try:
        return pd.read_csv(dataset_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Streamlit Interface
st.title("Resume Matcher with Skills Extraction and Job Dataset Integration")
st.subheader("Upload your resumes and job description to find the best match!")

# Load dataset
dataset_path = "jobs_dataset.csv"  # Ensure this file is in the same directory as your script
try:
    job_dataset = load_job_dataset(dataset_path)
    if job_dataset is not None:
        st.write("Job Titles Dataset Loaded Successfully!")
        
        # Display job title dropdown
        job_title = st.selectbox("Select a Job Title:", job_dataset["jobtitle"].unique(), key="job_title_selectbox")

        # Preload the job description into the text box
        if job_title:
            selected_job_description = job_dataset[job_dataset["jobtitle"] == job_title]["jobdescription"].iloc[0]
            # Display job description only in the text box
            st.session_state["job_description"] = st.text_area(
                "Preloaded Job Description:", value=selected_job_description, key="job_description_textarea"
            )
    else:
        st.error("Failed to load Job Titles Dataset. Please check the file format.")
except Exception as e:
    st.error(f"Error loading Jobs Dataset: {e}")

# Resume Upload
resume_files = st.file_uploader("Upload Resumes (PDFs):", type=["pdf"], accept_multiple_files=True)

if st.button("Match Resumes"):
    # Debugging outputs (optional)
    st.write(f"Uploaded Resumes: {[file.name for file in resume_files] if resume_files else 'None'}")

    if "job_description" in st.session_state and resume_files:
        # Preprocess Job Description and Resumes
        job_description = st.session_state["job_description"]
        preprocessed_job_desc = preprocess(job_description)
        preprocessed_resumes = [preprocess(extract_text_from_pdf(file)) for file in resume_files]

        # Extract and Display Skills
        job_skills = extract_skills(preprocessed_job_desc)
        st.write(f"**Skills Extracted from Job Description:** {', '.join(job_skills)}")

        if sbert_available:
            # Calculate SBERT Similarity
            scores = calculate_similarity_sbert(preprocessed_job_desc, preprocessed_resumes)

            # Display Similarity Scores for All Resumes
            st.subheader("Similarity Scores for All Resumes")
            similarity_results = sorted(
                zip([file.name for file in resume_files], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            for name, score in similarity_results:
                st.write(f"**Resume:** {name} | **Similarity Score:** {round(score * 100, 2)}%")

            # Top Match Explainability
            st.subheader("Explainability: Top Resume Match")
            best_name, best_score = similarity_results[0]
            best_resume = preprocessed_resumes[[file.name for file in resume_files].index(best_name)]

            st.write(f"**Top Matched Resume:** {best_name}")
            st.write(f"**Similarity Score:** {round(best_score * 100, 2)}%")

            # Explain Skills Matching
            resume_skills = extract_skills(best_resume)
            matched_skills = job_skills.intersection(resume_skills)
            missing_skills = job_skills.difference(resume_skills)

            st.write(f"**Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
            st.write(f"**Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")
        else:
            st.error("SBERT model is not available. Please ensure it is loaded correctly.")
    else:
        # Enhanced error messages
        if "job_description" not in st.session_state:
            st.error("Job description is missing. Please select a job title or provide a job description.")
        if not resume_files:
            st.error("No resumes uploaded. Please upload one or more resumes!")




#Functionality for Evaluation Metrics        
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# Load Gold Standard Dataset
@st.cache_data
def load_gold_standard(gold_standard_path):
    """
    Load the gold standard dataset from the provided CSV path.
    """
    return pd.read_csv(gold_standard_path)

# Calculate Precision, Recall, F1
def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate precision, recall, and F1 score.
    """
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    return precision, recall, f1

# Evaluate Model
def evaluate_predictions(similarity_scores, gold_standard_df, threshold=0.7):
    """
    Evaluate predictions against the gold standard and compute metrics.
    Args:
        similarity_scores (dict): A dictionary mapping job descriptions to similarity scores.
        gold_standard_df (pd.DataFrame): The gold standard DataFrame with 'Relevance'.
        threshold (float): Threshold for determining relevance.
    Returns:
        precision (float), recall (float), f1 (float)
    """
    # Generate Predicted Relevance based on threshold
    gold_standard_df['Predicted_Relevance'] = gold_standard_df['Job Description'].apply(
        lambda desc: 1 if similarity_scores.get(desc, 0) >= threshold else 0
    )

    # Check for imbalance
    if gold_standard_df['Relevance'].value_counts().min() < 5:
        st.warning("Gold standard dataset is imbalanced. Precision/Recall might not be accurate.")

    true_labels = gold_standard_df['Relevance']
    predicted_labels = gold_standard_df['Predicted_Relevance']

    # Calculate metrics
    precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)
    return precision, recall, f1

# Streamlit UI for Evaluation
st.header("Evaluation Metrics")
gold_standard_path = "gold_standard.csv"  # Ensure this file is in the same directory as your script
try:
    gold_standard_df = load_gold_standard(gold_standard_path)
    st.write("Gold Standard Dataset Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading Gold Standard Dataset: {e}")


if gold_standard_path:
    try:
        # Load gold standard dataset
        gold_standard_df = load_gold_standard(gold_standard_path)

        # Placeholder similarity scores (replace with your actual similarity calculations)
        similarity_scores = {
            desc: 0.8 if i % 2 == 0 else 0.6
            for i, desc in enumerate(gold_standard_df['Job Description'])
        }

        # Adjust threshold dynamically
        threshold = st.slider("Set Threshold for Relevance:", 0.5, 1.0, 0.8, 0.05)

        # Evaluate predictions
        precision, recall, f1 = evaluate_predictions(similarity_scores, gold_standard_df, threshold)

        # Display metrics
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

    except Exception as e:
        st.error(f"Error loading dataset: {e}")


