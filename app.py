from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import threading
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pypdf import PdfReader
import re
from flask_sqlalchemy import SQLAlchemy
from sentence_transformers import SentenceTransformer, util
import time
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-unsafe-key-for-dev')
upload_folder_name = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, upload_folder_name)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

# Database Configuration (Local PostgreSQL)
# Fallback updated to use 'mypswd' just in case the .env fails to load
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', "postgresql://app_user:Mindenszarhoz@localhost:5432/job_match_db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load NLP Model (Global) - Loaded once at startup
print("Loading NLP Model...")
nlp_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("NLP Model Loaded.")

# --- MODELS ---

class Users(UserMixin, db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    def get_id(self):
        return str(self.user_id)

class CVs(db.Model):
    __tablename__ = 'cvs'
    cv_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True) # Integer as per DB schema
    filename = db.Column(db.String(255), nullable=True) # Store the filename
    file_path = db.Column(db.String(512), nullable=True) # Store full path
    raw_text = db.Column(db.Text, nullable=False)
    parsed_tokens = db.Column(db.Text, nullable=True) # JSON or specific format if needed
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)

class Job_Descriptions(db.Model):
    __tablename__ = 'job_descriptions'
    jd_id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(255), nullable=True)
    title = db.Column(db.String(255), nullable=False)
    city = db.Column(db.String(255), nullable=True)
    country = db.Column(db.String(255), nullable=True)
    raw_text = db.Column(db.Text, nullable=False)
    url = db.Column(db.Text, nullable=True)
    parsed_tokens = db.Column(db.Text, nullable=True)
    is_intern = db.Column(db.Boolean, default=False)
    active_status = db.Column(db.Boolean, default=True)

class Precalc_Scores(db.Model):
    __tablename__ = 'precalc_scores'
    cv_id = db.Column(db.Integer, db.ForeignKey('cvs.cv_id'), primary_key=True)
    jd_id = db.Column(db.Integer, db.ForeignKey('job_descriptions.jd_id'), primary_key=True)
    similarity_score = db.Column(db.Float, nullable=False)

    cv = db.relationship('CVs', backref=db.backref('scores', cascade='all, delete-orphan'))
    job = db.relationship('Job_Descriptions', backref='scores')

@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

# --- HELPER FUNCTIONS ---

def clean_text(text):
    """
    Robust text cleaning.
    """
    if not text: return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_text_from_pdf(filepath):
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_text_from_txt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""

def calculate_matches_background(cv_id, cv_text):
    """
    Background task to calculate similarity scores for a new CV against all active JDs.
    """
    with app.app_context():
        print(f"Starting background matching for CV ID: {cv_id}")
        
        # 1. Fetch all active Job Descriptions
        active_jobs = Job_Descriptions.query.filter_by(active_status=True).all()
        if not active_jobs:
            print("No active jobs found for matching.")
            return

        job_texts = [job.raw_text for job in active_jobs] # Or use cleaned text if stored
        job_ids = [job.jd_id for job in active_jobs]

        # 2. Encode CV and Jobs
        cleaned_cv = clean_text(cv_text)
        cleaned_jobs = [clean_text(text) for text in job_texts]

        # Vectorize
        cv_embedding = nlp_model.encode(cleaned_cv, convert_to_tensor=True)
        job_embeddings = nlp_model.encode(cleaned_jobs, convert_to_tensor=True)

        # 3. Calculate Cosine Similarity
        scores = util.cos_sim(cv_embedding, job_embeddings)[0]

        # 4. Insert into Precalc_Scores
        new_scores = []
        for idx, score in enumerate(scores):
            similarity = float(score)
            new_scores.append(Precalc_Scores(
                cv_id=cv_id,
                jd_id=job_ids[idx],
                similarity_score=similarity
            ))
        
        # Batch insert for performance
        try:
            db.session.bulk_save_objects(new_scores)
            db.session.commit()
            print(f"Successfully calculated and saved {len(new_scores)} matches for CV ID: {cv_id}")
        except Exception as e:
            db.session.rollback()
            print(f"Error saving scores for CV ID {cv_id}: {e}")

def refresh_all_matches():
    """
    Recalculates match scores for ALL CVs against ALL active Job Descriptions.
    Used after bulk data updates.
    """
    with app.app_context():
        print("Starting full score refresh...")
        
        # 1. Fetch Data
        cvs = CVs.query.all()
        jobs = Job_Descriptions.query.filter_by(active_status=True).all()
        
        if not cvs or not jobs:
            print("Insufficient data (CVs or Jobs) to calculate matches.")
            return

        print(f"Matches to compute: {len(cvs)} CVs x {len(jobs)} Jobs")

        # 2. Prepare Embeddings
        # Note: Ideally caching these embeddings would be better, but for now we re-encode
        # to ensure consistency with any model updates or text cleaning changes.
        
        # Encode CVs
        cv_texts = [clean_text(cv.raw_text) for cv in cvs]
        cv_ids = [cv.cv_id for cv in cvs]
        cv_embeddings = nlp_model.encode(cv_texts, convert_to_tensor=True)

        # Encode Jobs
        job_texts = [clean_text(job.raw_text) for job in jobs]
        job_ids = [job.jd_id for job in jobs]
        job_embeddings = nlp_model.encode(job_texts, convert_to_tensor=True)

        # 3. Compute Similarity Matrix
        # util.cos_sim returns a matrix [num_cvs, num_jobs]
        cosine_scores = util.cos_sim(cv_embeddings, job_embeddings)

        # 4. Prepare Score Objects
        new_scores = []
        for i in range(len(cvs)):
            for j in range(len(jobs)):
                score = float(cosine_scores[i][j])
                # Store all scores to ensure Dream Job always has candidates
                new_scores.append(Precalc_Scores(
                    cv_id=cv_ids[i],
                    jd_id=job_ids[j],
                    similarity_score=score
                ))
        
        # 5. Update Database
        try:
            # Clear existing scores (Full Refresh)
            num_deleted = db.session.query(Precalc_Scores).delete()
            print(f"Deleted {num_deleted} old scores.")
            
            # Bulk Insert New
            db.session.bulk_save_objects(new_scores)
            db.session.commit()
            print(f"Successfully refreshed table with {len(new_scores)} matches.")
        except Exception as e:
            db.session.rollback()
            print(f"Error refreshing scores: {e}")

# --- ROUTES ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not email or not username or not password or not confirm_password:
            flash('All fields are required.')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))

        if Users.query.filter_by(email=email).first():
            flash('Email already exists.')
            return redirect(url_for('register'))
        
        if Users.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = Users(email=email, username=username, password_hash=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration.')
            print(e)
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = Users.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Show job board (listings)
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    pagination = Job_Descriptions.query.paginate(page=page, per_page=per_page, error_out=False)
    current_jobs = pagination.items
    
    return render_template('dashboard.html', jobs=current_jobs, page=page, total_pages=pagination.pages, has_next=pagination.has_next, has_prev=pagination.has_prev)

@app.route('/listings')
def listings():
    # Public (or Auth optional? User prompt said "Continue without logging in" goes to Job Board)
    # So we keep this route for non-logged in users.
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    pagination = Job_Descriptions.query.paginate(page=page, per_page=per_page, error_out=False)
    current_jobs = pagination.items
    
    return render_template('listings.html', jobs=current_jobs, page=page, total_pages=pagination.pages, has_next=pagination.has_next, has_prev=pagination.has_prev)


@app.route('/employer')
def employer():
    return "Employer specific portal - Coming Soon"

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_cvs = CVs.query.filter_by(user_id=current_user.user_id).order_by(CVs.upload_date.desc()).all()

    if request.method == 'POST':
        # Check CV Limit
        if len(user_cvs) >= 5:
            flash('You have reached the limit of 5 CVs. Please delete one to upload a new one.')
            return redirect(url_for('profile'))

        if 'resume' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['resume']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                # 1. Secure Filename
                filename = secure_filename(file.filename)
                unique_filename = f"{current_user.user_id}_{int(time.time())}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # 2. Save File
                file.save(filepath)
                
                # 3. Extract Text based on extension
                extracted_text = ""
                if filename.lower().endswith('.pdf'):
                    extracted_text = extract_text_from_pdf(filepath)
                elif filename.lower().endswith('.txt'):
                    extracted_text = extract_text_from_txt(filepath)
                
                # 4. Clean Text (Remove null bytes specifically)
                cleaned_text = clean_text(extracted_text)
                cleaned_text = cleaned_text.replace('\x00', '') # Extra safety
                
                # 5. Save to Database
                new_cv = CVs(
                    user_id=current_user.user_id,
                    filename=unique_filename,
                    file_path=filepath,
                    raw_text=cleaned_text # Store cleaned text to avoid DB errors
                )
                db.session.add(new_cv)
                db.session.commit()

                # 6. Trigger Background Match Calculation
                thread = threading.Thread(target=calculate_matches_background, args=(new_cv.cv_id, cleaned_text))
                thread.start()
                
                flash('Resume uploaded successfully! Job matching started in background.')
                return redirect(url_for('profile'))
            except Exception as e:
                db.session.rollback()
                print(f"Error during upload: {e}")
                flash(f'An error occurred during upload: {str(e)}')
                return redirect(request.url)

    return render_template('profile.html', user=current_user, cvs=user_cvs)

@app.route('/delete_cv/<int:cv_id>', methods=['POST'])
@login_required
def delete_cv(cv_id):
    cv = CVs.query.get_or_404(cv_id)
    
    # Ensure the CV belongs to the current user
    if cv.user_id != current_user.user_id:
        flash('Unauthorized action.')
        return redirect(url_for('profile'))
    
    try:
        # Delete file from filesystem
        if cv.file_path and os.path.exists(cv.file_path):
            os.remove(cv.file_path)
            
        # Delete from database
        Precalc_Scores.query.filter_by(cv_id=cv_id).delete()
        
        db.session.delete(cv)
        db.session.commit()
        flash('CV deleted successfully.')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting CV: {e}')
        
    return redirect(url_for('profile'))

@app.route('/dream_job')
@login_required
def dream_job():
    # Get latest CV
    user_cv = CVs.query.filter_by(user_id=current_user.user_id).order_by(CVs.upload_date.desc()).first()
    
    if not user_cv:
        flash("Please upload a resume first to see your dream job matches.")
        return redirect(url_for('profile'))

    # Get Top 3 Matches
    results = db.session.query(Precalc_Scores, Job_Descriptions).\
        join(Job_Descriptions, Precalc_Scores.jd_id == Job_Descriptions.jd_id).\
        filter(Precalc_Scores.cv_id == user_cv.cv_id).\
        order_by(Precalc_Scores.similarity_score.desc()).\
        limit(3).all() 

    matches = []
    for score_entry, job in results:
        matches.append({
            "id": job.jd_id,
            "title": job.title,
            "company": job.company,
            "description": job.raw_text[:300] + "...", 
            "score": round(score_entry.similarity_score * 100, 1),
            "url": job.url,
            "city": job.city,
            "country": job.country
        })

    return render_template('dream_jobs.html', matches=matches)


@app.route('/matches/<int:cv_id>')
def view_matches(cv_id):
    # Fast Read: Query Precalc_Scores joined with Job_Descriptions
    results = db.session.query(Precalc_Scores, Job_Descriptions).\
        join(Job_Descriptions, Precalc_Scores.jd_id == Job_Descriptions.jd_id).\
        filter(Precalc_Scores.cv_id == cv_id).\
        order_by(Precalc_Scores.similarity_score.desc()).\
        limit(50).all() 

    matches = []
    for score_entry, job in results:
        matches.append({
            "id": job.jd_id,
            "title": job.title,
            "company": job.company,
            "description": job.raw_text[:200] + "...", 
            "score": round(score_entry.similarity_score * 100, 2),
            "url": job.url
        })

    return render_template('job_seeker.html', matches=matches, cv_id=cv_id) 

# Initialize DB
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
