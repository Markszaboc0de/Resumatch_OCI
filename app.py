from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import threading
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import fitz # PyMuPDF
import re
from flask_sqlalchemy import SQLAlchemy
from sentence_transformers import SentenceTransformer, util
import time
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import json
import io
import html
from gliner import GLiNER

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
print("Loading NLP & NER Models...")
nlp_model = SentenceTransformer("BAAI/bge-m3")
ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
print("Models Loaded.")

# Global state to track active calculations per user
active_calculations = set()

# --- MODELS ---

class Users(UserMixin, db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    short_description = db.Column(db.String(500), nullable=True)
    is_visible = db.Column(db.Boolean, default=True)
    last_active_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    match_limit = db.Column(db.Integer, nullable=True, default=10)
    matches_used_this_month = db.Column(db.Integer, default=0)
    match_reset_date = db.Column(db.DateTime, nullable=True)
    
    supersearch_limit = db.Column(db.Integer, nullable=True, default=5)
    supersearch_used_this_month = db.Column(db.Integer, default=0)
    supersearch_reset_date = db.Column(db.DateTime, nullable=True)
    
    def get_id(self):
        return str(self.user_id)

    # Relationship to Notifications
    notifications = db.relationship('Notifications', backref='user', lazy=True)

class CVs(db.Model):
    __tablename__ = 'cvs'
    cv_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True) # Integer as per DB schema
    filename = db.Column(db.String(255), nullable=True) # Store the filename
    file_path = db.Column(db.String(512), nullable=True) # Store full path
    raw_text = db.Column(db.Text, nullable=False)
    parsed_tokens = db.Column(db.Text, nullable=True) # JSON or specific format if needed
    extracted_skills = db.Column(db.Text, nullable=True) # JSON list of extracted NER skills
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_main = db.Column(db.Boolean, default=False)

class Employers(db.Model):
    __tablename__ = 'employers'
    employer_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    company_name = db.Column(db.String(255), nullable=False)
    contact_info = db.Column(db.Text, nullable=True)
    
    match_limit = db.Column(db.Integer, nullable=True, default=10)
    matches_used_this_month = db.Column(db.Integer, default=0)
    match_reset_date = db.Column(db.DateTime, nullable=True)

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
    extracted_skills = db.Column(db.Text, nullable=True) # JSON list of extracted NER skills
    is_intern = db.Column(db.Boolean, default=False)
    active_status = db.Column(db.Boolean, default=True)
    is_native = db.Column(db.Boolean, default=False)
    employer_id = db.Column(db.Integer, db.ForeignKey('employers.employer_id'), nullable=True)
    
    sync_active = db.Column(db.Boolean, default=False)
    is_direct_upload = db.Column(db.Boolean, default=True)

    employer = db.relationship('Employers', backref=db.backref('jobs', lazy=True))

class Precalc_Scores(db.Model):
    __tablename__ = 'precalc_scores'
    cv_id = db.Column(db.Integer, db.ForeignKey('cvs.cv_id'), primary_key=True)
    jd_id = db.Column(db.Integer, db.ForeignKey('job_descriptions.jd_id'), primary_key=True)
    similarity_score = db.Column(db.Float, nullable=False)

    cv = db.relationship('CVs', backref=db.backref('scores', cascade='all, delete-orphan'))
    job = db.relationship('Job_Descriptions', backref='scores')

class Notifications(db.Model):
    __tablename__ = 'notifications'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    employer_id = db.Column(db.Integer, db.ForeignKey('employers.employer_id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job_descriptions.jd_id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Admins(db.Model):
    __tablename__ = 'admins'
    admin_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

class Surveys(db.Model):
    __tablename__ = 'surveys'
    survey_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), unique=True, nullable=False)
    survey_name = db.Column(db.String(255), nullable=False)
    survey_description = db.Column(db.String(500), nullable=False)
    survey_url = db.Column(db.Text, nullable=False)
    estimated_minutes = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('Users', backref=db.backref('survey', uselist=False, lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

# --- HELPER FUNCTIONS ---

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_text(text):
    """
    Robust text cleaning.
    """
    if not text: return ""
    text = html.unescape(text)
    
    # Remove script and style tags and their contents
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    # Use \w to keep all Unicode letters and numbers, replacing punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_skills_from_text(text):
    if not text: return "[]"
    try:
        labels = ["Skill", "Tool", "Technology", "Framework", "Software"]
        # Max length to avoid massive memory usage or slow inference
        ents = ner_model.predict_entities(text[:3000].lower(), set(labels))
        # Deduplicate
        skills = {ent['text'].strip() for ent in ents if len(ent['text'].strip()) >= 2}
        return json.dumps(list(skills))
    except Exception as e:
        print(f"NER Error: {e}")
        return "[]"

def extract_match_reasons(cv_skills_json, jd_skills_json):
    """
    Extracts the top overlapping skills between a CV and a Job Description.
    """
    if not cv_skills_json or not jd_skills_json:
        return ["No exact skill matches found"]
        
    try:
        cv_skills = set(json.loads(cv_skills_json))
        jd_skills = set(json.loads(jd_skills_json))
        
        overlap = cv_skills.intersection(jd_skills)
        if not overlap:
            return ["No exact skill matches found"]
            
        return list(overlap)[:3]
    except Exception:
        return ["No exact skill matches found"]

def extract_text_from_pdf(filepath):
    try:
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text() or ""
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

def check_and_deduct_match(entity):
    import datetime as dt
    now = dt.datetime.utcnow()
    
    # 1. Reset check
    if entity.match_reset_date is None or now >= entity.match_reset_date:
        entity.matches_used_this_month = 0
        entity.match_reset_date = now + dt.timedelta(days=30)
        db.session.commit()
        
    # 2. Limit check
    if entity.match_limit is None: # Infinite
        return True
        
    if entity.matches_used_this_month < entity.match_limit:
        entity.matches_used_this_month += 1
        db.session.commit()
        return True
        
    return False

def check_and_deduct_supersearch(entity):
    import datetime as dt
    now = dt.datetime.utcnow()
    
    # 1. Reset check
    if entity.supersearch_reset_date is None or now >= entity.supersearch_reset_date:
        entity.supersearch_used_this_month = 0
        entity.supersearch_reset_date = now + dt.timedelta(days=30)
        db.session.commit()
        
    # 2. Limit check
    if entity.supersearch_limit is None: # Infinite
        return True
        
    if entity.supersearch_used_this_month < entity.supersearch_limit:
        entity.supersearch_used_this_month += 1
        db.session.commit()
        return True
        
    return False

def load_cached_embeddings():
    """Helper to load pre-calculated job embeddings"""
    try:
        import torch
        cache_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}...")
            cache = torch.load(cache_path)
            return cache.get('embeddings'), cache.get('ids')
    except Exception as e:
        print(f"Failed to load cached embeddings: {e}")
    return None, None

def ensure_jit_matches(user):
    """
    Checks if the cache has been updated since the user was last active,
    or if the cache is missing. If so, triggers background recalculation.
    """
    try:
        if user.user_id in active_calculations:
            return
            
        cache_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
        needs_recalc = False
        
        if not os.path.exists(cache_path):
            needs_recalc = True
        else:
            cache_mtime = os.path.getmtime(cache_path)
            cache_time = datetime.utcfromtimestamp(cache_mtime)
            if not user.last_active_date or user.last_active_date < cache_time:
                needs_recalc = True
                
        if needs_recalc:
            print(f"JIT Matching trigger: Re-evaluating CVs for User {user.user_id}")
            active_calculations.add(user.user_id)
            user_cvs = CVs.query.filter_by(user_id=user.user_id).all()
            user_id_val = user.user_id
            
            def run_jit(u_id, cvs):
                try:
                    for cv in cvs:
                        if cv.parsed_tokens:
                            vec = json.loads(cv.parsed_tokens)
                            calculate_matches_background(cv.cv_id, vec)
                finally:
                    active_calculations.discard(u_id)
            
            thread = threading.Thread(target=run_jit, args=(user_id_val, user_cvs))
            thread.start()
            
            # Prevent infinite recalculation loop by immediately updating last_active_date
            user.last_active_date = datetime.utcnow()
            db.session.commit()
    except Exception as e:
        print(f"JIT matching evaluation failed: {e}")
        active_calculations.discard(user.user_id)

def calculate_matches_background(cv_id, cv_embedding_list):
    with app.app_context():
        import torch
        start_time = time.time()
        print(f"[{start_time}] Starting background matching for CV ID: {cv_id}")
        
        # 1. Check for Cached Embeddings
        job_embeddings, job_ids = load_cached_embeddings()
        
        if job_embeddings is None:
            print(f"[{time.time()}] Cache miss! Rebuilding cache from DB vectors...")
            active_jobs = Job_Descriptions.query.filter_by(active_status=True).all()
            if not active_jobs:
                print("No active jobs found for matching.")
                return 

            job_embeddings_list = []
            job_ids = []
            for job in active_jobs:
                if job.parsed_tokens and job.parsed_tokens.startswith('['):
                    try:
                        job_embeddings_list.append(json.loads(job.parsed_tokens))
                        job_ids.append(job.jd_id)
                    except:
                        pass
            
            if not job_embeddings_list:
                print("No valid encoded jobs found in DB! Make sure you run recalculate.py first.")
                return
                
            job_embeddings = torch.tensor(job_embeddings_list)
            
            try:
                cache_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
                torch.save({'embeddings': job_embeddings, 'ids': job_ids}, cache_path)
                print(f"[{time.time()}] Successfully rebuilt job_embeddings.pt cache with {len(job_ids)} jobs.")
            except Exception as e:
                print(f"Failed to save temporary cache: {e}")
        else:
            print(f"[{time.time()}] Loaded cached embeddings for {len(job_ids)} jobs (FAST).")

        # 2. Encode CV
        print(f"[{time.time()}] Using Pre-encoded User CV...")
        cv_embedding = torch.tensor(cv_embedding_list).unsqueeze(0) # [1, vector_dim]

        # 3. Calculate Cosine Similarity
        # cv_embedding is [1, vector_dim], job_embeddings is [N, vector_dim]
        # util.cos_sim returns [1, N]
        scores = util.cos_sim(cv_embedding, job_embeddings)[0]

        # Fetch skills for Hybrid Scoring
        cv = CVs.query.get(cv_id)
        cv_skills = set(json.loads(cv.extracted_skills or "[]")) if cv else set()
        active_jobs_for_skills = Job_Descriptions.query.filter(Job_Descriptions.jd_id.in_(job_ids)).all()
        job_skills_map = {job.jd_id: set(json.loads(job.extracted_skills or "[]")) for job in active_jobs_for_skills}

        # 4. Prepare Scores List
        all_scores = []
        for idx, score in enumerate(scores):
            semantic_score = float(score)
            jd_id = job_ids[idx]
            
            job_skills = job_skills_map.get(jd_id, set())
            if len(job_skills) == 0:
                final_score = semantic_score
            else:
                overlap = cv_skills.intersection(job_skills)
                ner_score = len(overlap) / len(job_skills)
                final_score = (semantic_score * 0.7) + (ner_score * 0.3)
                
            all_scores.append({
                'cv_id': cv_id,
                'jd_id': jd_id,
                'similarity_score': final_score
            })
            
        # 5. SORT and KEEP TOP 3 ONLY
        # This drastically reduces DB writes
        top_matches = sorted(all_scores, key=lambda x: x['similarity_score'], reverse=True)[:3]

        # 6. Insert into Precalc_Scores
        new_scores_objects = [Precalc_Scores(**match) for match in top_matches]
        
        # Batch insert for performance
        try:
            # Clear existing scores for this CV to avoid duplicates if re-running
            Precalc_Scores.query.filter_by(cv_id=cv_id).delete()
            
            db.session.bulk_save_objects(new_scores_objects)
            db.session.commit()
            print(f"Successfully calculated and saved Top {len(new_scores_objects)} matches for CV ID: {cv_id}")
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
        import torch
        cv_embeddings_list = []
        cv_ids = []
        
        for cv in cvs:
            if cv.parsed_tokens and cv.parsed_tokens.startswith('['):
                try:
                    vec = json.loads(cv.parsed_tokens)
                    cv_embeddings_list.append(vec)
                    cv_ids.append(cv.cv_id)
                except:
                    pass
            elif cv.raw_text: # Fallback to encoding old texts if necessary
                vec = nlp_model.encode(clean_text(cv.raw_text), convert_to_tensor=False)
                cv_embeddings_list.append(vec.tolist())
                cv_ids.append(cv.cv_id)
                
        if not cv_embeddings_list:
            print("No valid CV vectors found.")
            return
            
        cv_embeddings = torch.tensor(cv_embeddings_list)

        # Load Job Embeddings from DB
        job_embeddings_list = []
        job_ids = []
        for job in jobs:
            if job.parsed_tokens and job.parsed_tokens.startswith('['):
                try:
                    job_embeddings_list.append(json.loads(job.parsed_tokens))
                    job_ids.append(job.jd_id)
                except:
                    pass
                    
        if not job_embeddings_list:
            print("No valid encoded jobs found! Run recalculate.py first.")
            return
            
        job_embeddings = torch.tensor(job_embeddings_list)
        
        try:
            cache_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
            torch.save({'embeddings': job_embeddings, 'ids': job_ids}, cache_path)
            print("Saved updated job embeddings to cache.")
        except Exception as e:
            print(f"Failed to save updated job cache: {e}")

        # 3. Compute Similarity Matrix
        # 3. Calculate Scores Matrix [N_CVs, M_Jobs]
        # Only if both have items
        if len(cv_embeddings) > 0 and len(job_embeddings) > 0:
            all_scores_matrix = util.cos_sim(cv_embeddings, job_embeddings)
            
            # Fetch skills map for fast memory lookup
            job_skills_map = {job.jd_id: set(json.loads(job.extracted_skills or "[]")) for job in jobs}
            
            # 4. Save to DB (Optimized)
            new_scores_objects = []
            
            for i, cv in enumerate(cvs):
                cv_scores = all_scores_matrix[i]
                cv_skills = set(json.loads(cv.extracted_skills or "[]"))
                
                cv_matches = []
                for j, score in enumerate(cv_scores):
                    semantic_score = float(score)
                    jd_id = job_ids[j]
                    
                    job_skills = job_skills_map.get(jd_id, set())
                    if len(job_skills) == 0:
                        final_score = semantic_score
                    else:
                        overlap = cv_skills.intersection(job_skills)
                        ner_score = len(overlap) / len(job_skills)
                        final_score = (semantic_score * 0.7) + (ner_score * 0.3)
                        
                    cv_matches.append({
                        'cv_id': cv.cv_id,
                        'jd_id': jd_id,
                        'similarity_score': final_score
                    })
                
                # Sort and Keep Top 3
                top_10 = sorted(cv_matches, key=lambda x: x['similarity_score'], reverse=True)[:3]
                
                for match in top_10:
                    new_scores_objects.append(Precalc_Scores(**match))

            # Batch Insert
            try:
                # Truncate table first to be clean (optional but good for refresh)
                db.session.query(Precalc_Scores).delete()
                
                # Bulk save
                db.session.bulk_save_objects(new_scores_objects)
                db.session.commit()
                print(f"Full Refresh Logic Complete. Saved {len(new_scores_objects)} top matches.")
            except Exception as e:
                db.session.rollback()
                print(f"Error saving refresh matches: {e}")

                print(f"Error saving refresh matches: {e}")

# --- EMPLOYER ROUTES ---
from flask import session

@app.route('/employer/login', methods=['GET', 'POST'])
def employer_login():
    if 'employer_id' in session:
        return redirect(url_for('employer_dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        employer = Employers.query.filter_by(username=username).first()
        
        if employer and check_password_hash(employer.password_hash, password):
            session['employer_id'] = employer.employer_id
            session['employer_name'] = employer.company_name # Store separate from user
            flash(f'Welcome, {employer.company_name}!')
            return redirect(url_for('employer_dashboard'))
        else:
            flash('Invalid employer credentials.')
            return redirect(url_for('employer_login'))
            
    return render_template('employer_login.html')

@app.route('/employer/register', methods=['GET', 'POST'])
def employer_register():
    # Helper route to create an employer
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        company_name = request.form.get('company_name')
        
        if Employers.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('employer_register'))
            
        hashed_password = generate_password_hash(password)
        new_employer = Employers(username=username, password_hash=hashed_password, company_name=company_name)
        db.session.add(new_employer)
        db.session.commit()
        flash('Employer registered. Please log in.')
        return redirect(url_for('employer_login'))
    return render_template('employer_register.html') # Helper template

@app.route('/employer/logout')
def employer_logout():
    session.pop('employer_id', None)
    session.pop('employer_name', None)
    flash('Logged out from Employer Portal.')
    return redirect(url_for('landing'))

@app.route('/employer/dashboard')
def employer_dashboard():
    employer_id = session.get('employer_id')
    if not employer_id:
        flash('Please log in as an employer.')
        return redirect(url_for('employer_login'))
        
    employer = Employers.query.get(employer_id)
    jobs = Job_Descriptions.query.filter_by(employer_id=employer_id).order_by(Job_Descriptions.jd_id.desc()).all()
    
    return render_template('employer_dashboard.html', employer=employer, jobs=jobs)

@app.route('/employer/update_profile', methods=['POST'])
def employer_update_profile():
    employer_id = session.get('employer_id')
    if not employer_id:
        return redirect(url_for('employer_login'))
        
    employer = Employers.query.get(employer_id)
    if employer:
        employer.company_name = request.form.get('company_name')
        employer.contact_info = request.form.get('contact_info')
        db.session.commit()
        session['employer_name'] = employer.company_name # Update session name
        flash('Company profile updated.')
        
    return redirect(url_for('employer_dashboard'))

@app.route('/employer/create_job', methods=['POST'])
def create_job():
    employer_id = session.get('employer_id')
    if not employer_id:
        return redirect(url_for('employer_login'))
        
    title = request.form.get('title')
    city = request.form.get('city')
    country = request.form.get('country')
    description = request.form.get('description')
    url = request.form.get('url')
    apply_method = request.form.get('apply_method', 'redirect')
    
    is_native = (apply_method == 'native')
    
    # Ensure URL has protocol
    if url and not is_native and not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    elif is_native:
        url = None
    
    # Auto-fill company name
    employer = Employers.query.get(employer_id)
    company = employer.company_name
    
    # NLP Processing
    parsed_tokens = clean_text(description)
    
    new_job = Job_Descriptions(
        title=title,
        city=city,
        country=country,
        company=company,
        raw_text=description,
        url=url,
        parsed_tokens=parsed_tokens,
        employer_id=employer_id,
        active_status=True,
        is_native=is_native
    )
    
    try:
        db.session.add(new_job)
        db.session.commit()
        flash('Job posted successfully!')
    except Exception as e:
        db.session.rollback()
        flash(f'Error posting job: {e}')
        
    return redirect(url_for('employer_dashboard'))

@app.route('/employer/toggle_status/<int:job_id>')
def toggle_job_status(job_id):
    employer_id = session.get('employer_id')
    if not employer_id: return redirect(url_for('employer_login'))
    
    job = Job_Descriptions.query.get_or_404(job_id)
    if job.employer_id != employer_id:
        flash('Unauthorized.')
        return redirect(url_for('employer_dashboard'))
        
    job.active_status = not job.active_status
    db.session.commit()
    return redirect(url_for('employer_dashboard'))

@app.route('/employer/delete_job/<int:job_id>')
def delete_job(job_id):
    employer_id = session.get('employer_id')
    if not employer_id: return redirect(url_for('employer_login'))
    
    job = Job_Descriptions.query.get_or_404(job_id)
    if job.employer_id != employer_id:
        flash('Unauthorized.')
        return redirect(url_for('employer_dashboard'))
        
    # Also delete associated scores? Yes, cascaded in DB or manual
    # Precalc_Scores has foreign key to jd_id, need to check cascade rule or delete manually
    Precalc_Scores.query.filter_by(jd_id=job_id).delete()
    
    # Also delete associated notifications
    Notifications.query.filter_by(job_id=job_id).delete()
    
    db.session.delete(job)
    db.session.commit()
    flash('Job deleted.')
    return redirect(url_for('employer_dashboard'))

@app.route('/employer/match/<int:job_id>')
def employer_match_candidate(job_id):
    employer_id = session.get('employer_id')
    if not employer_id: return redirect(url_for('employer_login'))
    
    job = Job_Descriptions.query.get_or_404(job_id)
    if job.employer_id != employer_id:
        flash('Unauthorized.')
        return redirect(url_for('employer_dashboard'))
        
    employer = Employers.query.get(employer_id)
    if not check_and_deduct_match(employer):
        flash('You have reached your matching limit for this month.')
        return redirect(url_for('employer_dashboard'))
    
    keywords_param = request.args.get('keywords', '')
    filter_keywords = [kw.strip().lower() for kw in keywords_param.split(',') if kw.strip()]
    
    # 1. Get Job Text/Embedding
    job_text = clean_text(job.raw_text)
    job_embedding = nlp_model.encode(job_text, convert_to_tensor=True)
    
    # 2. Get All Candidates
    import datetime as dt
    seven_days_ago = dt.datetime.utcnow() - dt.timedelta(days=7)
    all_cvs = db.session.query(CVs).join(Users, CVs.user_id == Users.user_id).filter(
        CVs.user_id.isnot(None), 
        Users.is_visible == True,
        Users.last_active_date >= seven_days_ago
    ).all()
    
    # Filter by keywords if provided
    cvs = []
    if filter_keywords:
        for cv in all_cvs:
            cv_raw_lower = cv.raw_text.lower() if cv.raw_text else ""
            # Must match at least one keyword/variation
            if any(kw in cv_raw_lower for kw in filter_keywords):
                cvs.append(cv)
    else:
        cvs = all_cvs

    if not cvs:
        flash('No visible candidates found matching the criteria.')
        return redirect(url_for('employer_dashboard'))
        
    import torch
    valid_cvs = []
    cv_embeddings_list = []
    
    for cv in cvs:
        if cv.parsed_tokens and cv.parsed_tokens.startswith('['):
            try:
                vec = json.loads(cv.parsed_tokens)
                cv_embeddings_list.append(vec)
                valid_cvs.append(cv)
            except:
                pass
        elif cv.raw_text:
            vec = nlp_model.encode(clean_text(cv.raw_text), convert_to_tensor=False)
            cv_embeddings_list.append(vec.tolist())
            valid_cvs.append(cv)
            
    if not valid_cvs:
        flash('No valid candidates found (no vectors).')
        return redirect(url_for('employer_dashboard'))
        
    cvs = valid_cvs
    cv_embeddings = torch.tensor(cv_embeddings_list)
    
    # 3. Compute Similarity
    # job_embedding: [1, 384], cv_embeddings: [N, 384]
    scores = util.cos_sim(job_embedding, cv_embeddings)[0] # [N]
    
    # 4. Find Top 5 Matches
    all_scores = [(float(score), i) for i, score in enumerate(scores)]
    all_scores.sort(key=lambda x: x[0], reverse=True)
    
    top_matches = all_scores[:5]
    
    match_list = []
    if top_matches:
        for score_val, cv_idx in top_matches:
            cv_item = cvs[cv_idx]
            match_percentage = round(score_val * 100, 1)
            
            # Get User details
            candidate_user = Users.query.get(cv_item.user_id)
            
            # Extract Match Reasons
            reasons = extract_match_reasons(cv_item.extracted_skills, job.extracted_skills)
            
            # Check if already notified
            has_notified = False
            if Notifications.query.filter_by(user_id=candidate_user.user_id, employer_id=employer_id, job_id=job_id).first():
                has_notified = True

            match_list.append({
                'candidate': candidate_user,
                'score': match_percentage,
                'cv': cv_item,
                'reasons': reasons,
                'has_notified': has_notified
            })

        return render_template('candidate_match.html', 
                               job=job, 
                               matches=match_list,
                               active_keywords=keywords_param)
    else:
        flash('Could not determine any matches.')
    return redirect(url_for('employer_dashboard'))

@app.route('/employer/notify/<int:job_id>/<int:candidate_id>', methods=['POST'])
def notify_candidate(job_id, candidate_id):
    employer_id = session.get('employer_id')
    if not employer_id:
        flash('Please log in.')
        return redirect(url_for('employer_login'))
    
    # 1. Check for duplicates
    existing = Notifications.query.filter_by(
        user_id=candidate_id, 
        employer_id=employer_id, 
        job_id=job_id
    ).first()
    
    if existing:
        flash('You have already notified this candidate for this job.')
        return redirect(url_for('employer_match_candidate', job_id=job_id))
        
    # 2. Get Details for Message
    employer = Employers.query.get(employer_id)
    job = Job_Descriptions.query.get(job_id)
    # custom_message removed in favor of persistent contact_info
    
    # 3. Create Notification
    base_msg = f"Good news! {employer.company_name} is interested in your profile for the {job.title} position."
    
    if employer.contact_info:
        msg = f"{base_msg}\n\nContact Details:\n{employer.contact_info}"
    else:
        msg = f"{base_msg} Please contact them."
    
    new_notif = Notifications(
        user_id=candidate_id,
        employer_id=employer_id,
        job_id=job_id,
        message=msg
    )
    
    try:
        db.session.add(new_notif)
        db.session.commit()
        flash('Candidate notified!')
    except Exception as e:
        db.session.rollback()
        flash(f'Error sending notification: {e}')
        
    return redirect(url_for('employer_match_candidate', job_id=job_id))

@app.route('/notifications')
@login_required
def notifications():
    user_notifs = Notifications.query.filter_by(user_id=current_user.user_id).order_by(Notifications.timestamp.desc()).all()
    return render_template('notifications.html', notifications=user_notifs)

@app.route('/notifications/mark_read/<int:notification_id>')
@login_required
def mark_notification_read(notification_id):
    notif = Notifications.query.get_or_404(notification_id)
    if notif.user_id != current_user.user_id:
        flash('Unauthorized.')
        return redirect(url_for('notifications'))
        
    notif.is_read = True
    db.session.commit()
    return redirect(url_for('notifications'))

# --- ADMIN ROUTES ---
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if 'admin_id' in session:
        return redirect(url_for('admin_dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin = Admins.query.filter_by(username=username).first()
        
        if admin and check_password_hash(admin.password_hash, password):
            session['admin_id'] = admin.admin_id
            session['admin_username'] = admin.username
            flash(f'Welcome Admin {admin.username}!')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials.')
            return redirect(url_for('admin_login'))
            
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    session.pop('admin_username', None)
    flash('Logged out from Admin Portal.')
    return redirect(url_for('landing'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        flash('Please log in as an Admin.')
        return redirect(url_for('admin_login'))
        
    users = Users.query.all()
    employers = Employers.query.all()
    
    return render_template('admin_dashboard.html', users=users, employers=employers)

@app.route('/admin/edit_user/<int:user_id>', methods=['POST'])
def admin_edit_user(user_id):
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
        
    user = Users.query.get_or_404(user_id)
    
    user.is_visible = request.form.get('is_visible') == 'on'
    limit_val = request.form.get('match_limit')
    
    if not limit_val or limit_val.strip() == '' or limit_val.strip().lower() == 'infinite':
        user.match_limit = None
    else:
        try:
            user.match_limit = int(limit_val)
        except ValueError:
            pass # Or handle error
    
    try:
        user.matches_used_this_month = int(request.form.get('matches_used_this_month', user.matches_used_this_month))
    except ValueError:
        pass
        
    ss_limit_val = request.form.get('supersearch_limit')
    
    if not ss_limit_val or ss_limit_val.strip() == '' or ss_limit_val.strip().lower() == 'infinite':
        user.supersearch_limit = None
    else:
        try:
            user.supersearch_limit = int(ss_limit_val)
        except ValueError:
            pass
            
    try:
        user.supersearch_used_this_month = int(request.form.get('supersearch_used_this_month', user.supersearch_used_this_month))
    except ValueError:
        pass
    
    db.session.commit()
    flash(f'User {user.username} updated.')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/edit_employer/<int:employer_id>', methods=['POST'])
def admin_edit_employer(employer_id):
    if 'admin_id' not in session:
        return redirect(url_for('admin_login'))
        
    employer = Employers.query.get_or_404(employer_id)
    
    limit_val = request.form.get('match_limit')
    
    if not limit_val or limit_val.strip() == '' or limit_val.strip().lower() == 'infinite':
        employer.match_limit = None
    else:
        try:
            employer.match_limit = int(limit_val)
        except ValueError:
            pass
            
    try:
        employer.matches_used_this_month = int(request.form.get('matches_used_this_month', employer.matches_used_this_month))
    except ValueError:
        pass
        
    db.session.commit()
    flash(f'Employer {employer.company_name} updated.')
    return redirect(url_for('admin_dashboard'))

# Context Processor for Badge Count
@app.context_processor
def inject_notifications():
    if current_user.is_authenticated:
        unread_count = Notifications.query.filter_by(user_id=current_user.user_id, is_read=False).count()
        return dict(unread_count=unread_count)
    return dict(unread_count=0)

# --- ROUTES ---

@app.route('/job/<int:jd_id>')
def job_detail(jd_id):
    job = Job_Descriptions.query.get_or_404(jd_id)
    if not job.active_status:
        flash("This job is no longer active.")
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return redirect(url_for('listings'))
    return render_template('job_detail.html', job=job)

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
            
            # Just-In-Time (JIT) Match Calculation
            ensure_jit_matches(user)
                
            # Update last active timestamp so it doesn't infinite loop on next refresh
            # and prevents profile from expiring
            user.last_active_date = datetime.utcnow()
            db.session.commit()
            
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
    ensure_jit_matches(current_user)

    # Show job board (listings)
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Filter Parameters
    search_query = request.args.get('search', '')
    country_filter = request.args.get('country', '')
    city_filter = request.args.get('city', '')
    company_filter = request.args.get('company', '')

    query = Job_Descriptions.query.filter_by(active_status=True)

    if search_query:
        query = query.filter(Job_Descriptions.raw_text.ilike(f'%{search_query}%') | Job_Descriptions.title.ilike(f'%{search_query}%'))
    if country_filter:
        query = query.filter(Job_Descriptions.country.ilike(f'%{country_filter}%'))
    if city_filter:
        query = query.filter(Job_Descriptions.city.ilike(f'%{city_filter}%'))
    if company_filter:
        query = query.filter(Job_Descriptions.company.ilike(f'%{company_filter}%'))

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    current_jobs = pagination.items
    
    return render_template('dashboard.html', jobs=current_jobs, page=page, total_pages=pagination.pages, has_next=pagination.has_next, has_prev=pagination.has_prev,
                           search_query=search_query, country_filter=country_filter, city_filter=city_filter, company_filter=company_filter)

@app.route('/listings')
def listings():
    # Public (or Auth optional? User prompt said "Continue without logging in" goes to Job Board)
    # So we keep this route for non-logged in users.
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Filter Parameters
    search_query = request.args.get('search', '')
    country_filter = request.args.get('country', '')
    city_filter = request.args.get('city', '')
    company_filter = request.args.get('company', '')

    query = Job_Descriptions.query.filter_by(active_status=True)

    if search_query:
        query = query.filter(Job_Descriptions.raw_text.ilike(f'%{search_query}%') | Job_Descriptions.title.ilike(f'%{search_query}%'))
    if country_filter:
        query = query.filter(Job_Descriptions.country.ilike(f'%{country_filter}%'))
    if city_filter:
        query = query.filter(Job_Descriptions.city.ilike(f'%{city_filter}%'))
    if company_filter:
        query = query.filter(Job_Descriptions.company.ilike(f'%{company_filter}%'))

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    current_jobs = pagination.items
    
    return render_template('listings.html', jobs=current_jobs, page=page, total_pages=pagination.pages, has_next=pagination.has_next, has_prev=pagination.has_prev,
                           search_query=search_query, country_filter=country_filter, city_filter=city_filter, company_filter=company_filter)

@app.route('/map')
def job_map():
    from flask import session
    if 'employer_id' in session:
        return redirect(url_for('employer_dashboard'))
        
    # Only show active jobs. Use direct session query to bypass missing employer_id column crashes
    jobs = db.session.query(
        Job_Descriptions.jd_id, 
        Job_Descriptions.title, 
        Job_Descriptions.company, 
        Job_Descriptions.url, 
        Job_Descriptions.city, 
        Job_Descriptions.country
    ).filter(Job_Descriptions.active_status == True).all()
    
    # Load cache
    cache = {}
    cache_path = os.path.join(BASE_DIR, 'location_cache.json')
    if os.path.exists(cache_path):
        import json
        with open(cache_path, 'r') as f:
            try:
                cache = json.load(f)
            except:
                pass
                
    # Build Map Data
    map_data = {}
    total_mapped = 0
    
    for job in jobs:
        if not job.city:
            continue
            
        key = f"{job.city},{job.country}" if job.country else f"{job.city}"
        coords = cache.get(key)
        
        # Only map resolved coordinates
        if coords and isinstance(coords, dict) and 'lat' in coords and 'lon' in coords:
            if key not in map_data:
                map_data[key] = {
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'jobs': []
                }
                
            map_data[key]['jobs'].append({
                'id': job.jd_id,
                'title': job.title,
                'company': job.company,
                'url': job.url
            })
            total_mapped += 1
            
    import json
    return render_template('map.html', map_data=json.dumps(map_data), total_mapped_jobs=total_mapped)


@app.route('/employer')
def employer():
    return "Employer specific portal - Coming Soon"

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    ensure_jit_matches(current_user)
    
    user_cvs = CVs.query.filter_by(user_id=current_user.user_id).order_by(CVs.upload_date.desc()).all()

    if request.method == 'POST':
        # Update Profile (Description + Visibility)
        if 'short_description' in request.form:
            current_user.short_description = request.form.get('short_description')
            current_user.is_visible = request.form.get('is_visible') == 'on'
            
            # Renew active status
            current_user.last_active_date = datetime.utcnow()
            
            db.session.commit()
            flash('Profile updated and active status renewed successfully.')
            return redirect(url_for('profile'))

        # Check CV Limit
        if len(user_cvs) >= 5:
            flash('You have reached the limit of 5 CVs. Please delete one to upload a new one.')
            return redirect(url_for('profile'))

        if 'resume' not in request.files:
            flash('No file part')
            return redirect(url_for('profile'))
        
        file = request.files['resume']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('profile'))
            
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 5 * 1024 * 1024:
            flash('CV file size exceeds the 5MB limit.')
            return redirect(url_for('profile'))
            
        if file and allowed_file(file.filename):
            try:
                # 1. Secure Filename
                filename = secure_filename(file.filename)
                unique_filename = f"{current_user.user_id}_{int(time.time())}_{filename}"
                
                # 2. Extract Text Memory
                file_bytes = file.read()
                extracted_text = ""
                if filename.lower().endswith('.pdf'):
                    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                        for page in doc:
                            extracted_text += page.get_text() or ""
                elif filename.lower().endswith('.txt'):
                    extracted_text = file_bytes.decode('utf-8', errors='ignore')
                
                # Strip null bytes that crash PostgreSQL
                extracted_text = extracted_text.replace('\x00', '')
                
                # 3. Clean Text, Encode, and Extract NER Skills
                cleaned_text = clean_text(extracted_text)
                cv_skills_json = extract_skills_from_text(cleaned_text)
                cv_embedding = nlp_model.encode(cleaned_text, convert_to_tensor=False)
                vector_json = json.dumps(cv_embedding.tolist())
                
                # 4. Save to Database
                new_cv = CVs(
                    user_id=current_user.user_id,
                    filename=unique_filename,
                    file_path=None,
                    raw_text=extracted_text,  # Save the actual text for match reasoning
                    parsed_tokens=vector_json,
                    extracted_skills=cv_skills_json
                )
                db.session.add(new_cv)
                db.session.commit()

                # 5. Trigger Background Match Calculation
                active_calculations.add(current_user.user_id)
                def run_upload_jit(u_id, c_id, vec):
                    try:
                        calculate_matches_background(c_id, vec)
                    finally:
                        active_calculations.discard(u_id)
                        
                thread = threading.Thread(target=run_upload_jit, args=(current_user.user_id, new_cv.cv_id, cv_embedding.tolist()))
                thread.start()
                
                flash('Resume uploaded successfully! Job matching started in background.')
                return redirect(url_for('profile'))
            except Exception as e:
                db.session.rollback()
                print(f"Error during upload: {e}")
                error_str = str(e)
                flash(f'An error occurred during upload: {error_str[:150]}...')
                return redirect(url_for('profile'))

    import datetime as dt
    seven_days_ago = dt.datetime.utcnow() - dt.timedelta(days=7)
    is_expired = False
    if current_user.last_active_date and current_user.last_active_date < seven_days_ago:
        is_expired = True

    user_survey = Surveys.query.filter_by(user_id=current_user.user_id).first()

    return render_template('profile.html', user=current_user, cvs=user_cvs, is_expired=is_expired, user_survey=user_survey)

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

@app.route('/set_main_cv/<int:cv_id>', methods=['POST'])
@login_required
def set_main_cv(cv_id):
    cv = CVs.query.get_or_404(cv_id)
    if cv.user_id != current_user.user_id:
        flash('Unauthorized action.')
        return redirect(url_for('profile'))
    
    try:
        # Reset all other CVs for this user
        CVs.query.filter_by(user_id=current_user.user_id).update({'is_main': False})
        
        # Set this one as main
        cv.is_main = True
        db.session.commit()
        flash('Main resume updated successfully.')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating main resume: {e}')
        
    return redirect(url_for('profile'))

@app.route('/dream_job')
@login_required
def dream_job():
    ensure_jit_matches(current_user)

    if not check_and_deduct_match(current_user):
        flash("You have reached your matching limit for this month.")
        return redirect(url_for('profile'))

    # 1. Try to get the "Main" CV
    user_cv = CVs.query.filter_by(user_id=current_user.user_id, is_main=True).first()
    
    # 2. Fallback: Get most recent CV
    if not user_cv:
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
@login_required
def view_matches(cv_id):
    cv = CVs.query.get_or_404(cv_id)
    if cv.user_id != current_user.user_id:
        flash('Unauthorized action.')
        return redirect(url_for('profile'))
        
    if not check_and_deduct_match(current_user):
        flash('You have reached your matching limit for this month.')
        return redirect(url_for('profile'))

    # Fast Read: Query Precalc_Scores joined with Job_Descriptions
    results = db.session.query(Precalc_Scores, Job_Descriptions).\
        join(Job_Descriptions, Precalc_Scores.jd_id == Job_Descriptions.jd_id).\
        filter(Precalc_Scores.cv_id == cv_id).\
        order_by(Precalc_Scores.similarity_score.desc()).\
        limit(50).all() 

    cv = CVs.query.get(cv_id)
    cv_text = cv.raw_text if cv else ""

    matches = []
    for score_entry, job in results:
        reasons = extract_match_reasons(cv.extracted_skills, job.extracted_skills)
        matches.append({
            "id": job.jd_id,
            "title": job.title,
            "company": job.company,
            "description": job.raw_text[:200] + "...", 
            "score": round(score_entry.similarity_score * 100, 2),
            "url": job.url,
            "reasons": reasons
        })

    return render_template('job_seeker.html', matches=matches, cv_id=cv_id) 

@app.route('/api/suggestions')
def suggestions():
    field = request.args.get('field')
    query = request.args.get('query', '')
    
    if not field or len(query) < 1:
        return jsonify([])
    
    column = None
    if field == 'city':
        column = Job_Descriptions.city
    elif field == 'country':
        column = Job_Descriptions.country
    elif field == 'company':
        column = Job_Descriptions.company
        
    if column:
        results = db.session.query(column).filter(column.ilike(f'%{query}%')).distinct().limit(10).all()
        # results is a list of tuples, e.g. [('London',), ('New York',)]
        suggestions = [r[0] for r in results if r[0]]
        return jsonify(suggestions)
    
    return jsonify([]) 

@app.route('/api/supersearch', methods=['POST'])
@login_required
def supersearch():
    if not check_and_deduct_supersearch(current_user):
        return jsonify({'error': 'You have reached your Supersearch limit for this month.'}), 403

    data = request.get_json()
    if not data or 'ideal_job_description' not in data:
        return jsonify({'error': 'Missing ideal_job_description'}), 400

    ideal_desc = data['ideal_job_description']
    if not ideal_desc.strip():
        return jsonify({'error': 'Description cannot be empty'}), 400

    # Clean text and encode
    cleaned_desc = clean_text(ideal_desc)
    ideal_embedding = nlp_model.encode(cleaned_desc, convert_to_tensor=True)

    # Calculate matches
    import torch
    job_embeddings, job_ids = load_cached_embeddings()
    active_jobs = Job_Descriptions.query.filter_by(active_status=True).all()
    
    if not active_jobs:
        return jsonify({'matches': [], 'remaining': 'Infinite' if current_user.supersearch_limit is None else current_user.supersearch_limit - current_user.supersearch_used_this_month})

    if job_embeddings is None:
        job_texts = [clean_text(job.raw_text) for job in active_jobs]
        job_ids = [job.jd_id for job in active_jobs]
        job_embeddings = nlp_model.encode(job_texts, convert_to_tensor=True)
        
        try:
            cache_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
            torch.save({'embeddings': job_embeddings, 'ids': job_ids}, cache_path)
        except Exception:
            pass
    
    scores = util.cos_sim(ideal_embedding, job_embeddings)[0]
    
    active_job_ids = {job.jd_id: job for job in active_jobs}
    
    all_scores = []
    for idx, score in enumerate(scores):
        if job_ids[idx] in active_job_ids:
            all_scores.append((float(score), job_ids[idx]))

    top_matches = sorted(all_scores, key=lambda x: x[0], reverse=True)[:3]

    matches_response = []
    for score, jd_id in top_matches:
        job = active_job_ids[jd_id]
        matches_response.append({
            "id": job.jd_id,
            "title": job.title,
            "company": job.company,
            "description": job.raw_text[:300] + "...",
            "score": round(score * 100, 1),
            "url": job.url,
            "city": job.city,
            "country": job.country
        })

    remaining = 'Infinite' if current_user.supersearch_limit is None else current_user.supersearch_limit - current_user.supersearch_used_this_month
    return jsonify({
        'matches': matches_response,
        'remaining': remaining
    })

@app.route('/api/match_status')
@login_required
def match_status():
    is_calculating = current_user.user_id in active_calculations
    return jsonify({'is_calculating': is_calculating})

# --- SURVEY ROUTES ---
@app.route('/survey')
def survey_board():
    surveys = Surveys.query.order_by(Surveys.created_at.desc()).all()
    return render_template('survey.html', surveys=surveys)

@app.route('/profile/survey', methods=['POST'])
@login_required
def save_survey():
    survey_name = request.form.get('survey_name', '').strip()
    survey_description = request.form.get('survey_description', '').strip()
    survey_url = request.form.get('survey_url', '').strip()
    estimated_minutes = request.form.get('estimated_minutes', '').strip()

    if not survey_name or not survey_description or not survey_url or not estimated_minutes:
        flash('All survey fields are required.')
        return redirect(url_for('profile'))

    # Ensure URL has protocol
    if not survey_url.startswith(('http://', 'https://')):
        survey_url = 'https://' + survey_url

    try:
        minutes = int(estimated_minutes)
        if minutes < 1 or minutes > 120:
            flash('Estimated time must be between 1 and 120 minutes.')
            return redirect(url_for('profile'))
    except ValueError:
        flash('Estimated time must be a number.')
        return redirect(url_for('profile'))

    existing = Surveys.query.filter_by(user_id=current_user.user_id).first()
    if existing:
        existing.survey_name = survey_name
        existing.survey_description = survey_description
        existing.survey_url = survey_url
        existing.estimated_minutes = minutes
    else:
        new_survey = Surveys(
            user_id=current_user.user_id,
            survey_name=survey_name,
            survey_description=survey_description,
            survey_url=survey_url,
            estimated_minutes=minutes
        )
        db.session.add(new_survey)

    db.session.commit()
    flash('Survey saved successfully!')
    return redirect(url_for('profile'))

@app.route('/profile/survey/delete', methods=['POST'])
@login_required
def delete_survey():
    survey = Surveys.query.filter_by(user_id=current_user.user_id).first()
    if survey:
        db.session.delete(survey)
        db.session.commit()
        flash('Survey removed.')
    return redirect(url_for('profile'))

# Initialize DB
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
