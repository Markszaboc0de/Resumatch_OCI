from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
import re
import math
from collections import Counter
import pandas as pd

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global data storage
JOBS = []
RESUMES = []

def load_data():
    """Lengths data from CSV files into global lists."""
    global JOBS, RESUMES
    
    # Load Jobs
    jobs_path = os.path.join(BASE_DIR, 'jobs.csv')
    if os.path.exists(jobs_path):
        try:
            # Using semi-colon separator as seen in the file
            jobs_df = pd.read_csv(jobs_path, sep=';', on_bad_lines='skip', encoding='utf-8')
            # Look for columns. Adjust if names differ slightly in CSV
            # Based on previous file reads, columns seem to be: 'ID', 'Job Title', 'Job Description' etc.
            # We'll normalize column names just in case or use exact names
            for _, row in jobs_df.iterrows():
                # Flexible column access
                job_id = row.get('ID')
                title = row.get('Job Title')
                desc = row.get('Job Description')
                url = row.get('URL')
                
                if pd.notna(job_id) and pd.notna(title) and pd.notna(desc):
                    JOBS.append({
                        'id': int(job_id),
                        'title': str(title),
                        'description': str(desc),
                        'url': str(url) if pd.notna(url) else '#'
                    })
            print(f"Loaded {len(JOBS)} jobs from CSV.")
        except Exception as e:
            print(f"Error loading jobs.csv: {e}")
    else:
        print("jobs.csv not found.")

    # Load Resumes
    resumes_path = os.path.join(BASE_DIR, 'UpdatedResumeDataSet.csv')
    if os.path.exists(resumes_path):
        try:
            # Based on previous file reads, columns: 'ID', 'Category', 'Resume'
            resumes_df = pd.read_csv(resumes_path, sep=';', on_bad_lines='skip', encoding='utf-8')
            for _, row in resumes_df.iterrows():
                resume_id = row.get('ID')
                category = row.get('Category')
                text = row.get('Resume')
                
                if pd.notna(category) and pd.notna(text):
                     # Resume ID might key, if not present use index or generate one
                    if pd.isna(resume_id):
                        resume_id = len(RESUMES) + 1
                        
                    RESUMES.append({
                        'id': int(resume_id) if pd.notna(resume_id) else len(RESUMES) + 1,
                        'category': str(category),
                        'resume_text': str(text)
                    })
            print(f"Loaded {len(RESUMES)} resumes from CSV.")
        except Exception as e:
            print(f"Error loading UpdatedResumeDataSet.csv: {e}")
    else:
        print("UpdatedResumeDataSet.csv not found.")

# Load data on startup
load_data()


# --- TEXT PROCESSING & MATCHING LOGIC ---

def clean_text(text):
    """
    Tokenizes and cleans text: removes non-alphanumeric characters, converts to lower case,
    and removes common stopwords.
    Returns a list of words.
    """
    STOPWORDS = {
        'and', 'the', 'is', 'in', 'at', 'of', 'a', 'with', 'using', 'for', 'to', 'an', 'or', 'by', 'on'
    }
    # Ensure text is string
    if not isinstance(text, str):
        text = str(text)
        
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.lower().split()
    return [word for word in tokens if word not in STOPWORDS]

def get_cosine_similarity(text1, text2):
    """
    Calculates Cosine Similarity between two text strings.
    """
    tokens1 = clean_text(text1)
    tokens2 = clean_text(text2)

    vec1 = Counter(tokens1)
    vec2 = Counter(tokens2)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return numerator / denominator

def extract_text_from_pdf(filepath):
    """
    Extracts text from a PDF file using pdfminer.six.
    """
    try:
        return extract_text(filepath)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_text_from_txt(filepath):
    """
    Extracts text from a TXT file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/listings')
def listings():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    total_jobs = len(JOBS)
    total_pages = math.ceil(total_jobs / per_page)
    
    start = (page - 1) * per_page
    end = start + per_page
    
    current_jobs = JOBS[start:end]
    
    has_next = page < total_pages
    has_prev = page > 1
    
    return render_template('listings.html', jobs=current_jobs, page=page, total_pages=total_pages, has_next=has_next, has_prev=has_prev)

@app.route('/employer', methods=['GET', 'POST'])
def employer():
    matches = []
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.txt'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            job_desc_text = extract_text_from_txt(filepath)
            
            # Match against Resumes List
            scored_resumes = []
            for resume in RESUMES:
                score = get_cosine_similarity(job_desc_text, resume['resume_text'])
                scored_resumes.append({
                    "id": resume['id'],
                    "name": resume['category'], # Using category as name/identifier for now
                    "text": resume['resume_text'],
                    "score": round(score * 100, 2)
                })
            
            # Sort by score descending
            matches = sorted(scored_resumes, key=lambda x: x['score'], reverse=True)[:3]
            
            # Simple cleanup of uploaded file
            os.remove(filepath)
            
    return render_template('employer.html', matches=matches)

@app.route('/job_seeker', methods=['GET', 'POST'])
def job_seeker():
    matches = []
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            resume_text = extract_text_from_pdf(filepath)
            
            # Match against Jobs List
            scored_jobs = []
            for job in JOBS:
                score = get_cosine_similarity(resume_text, job['description'])
                scored_jobs.append({
                    "id": job['id'],
                    "title": job['title'],
                    "description": job['description'],
                    "url": job.get('url', '#'),
                    "score": round(score * 100, 2)
                })
            
            # Sort by score descending
            matches = sorted(scored_jobs, key=lambda x: x['score'], reverse=True)[:3]
            
            os.remove(filepath)

    return render_template('job_seeker.html', matches=matches)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # In development, use debug=True. In production (like pythonanywhere), 
    # the WSGI server will handle this, but debug=False is safer.
    app.run(host='0.0.0.0', port=port, debug=True)

