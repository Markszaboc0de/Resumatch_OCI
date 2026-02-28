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
    short_description = db.Column(db.String(500), nullable=True)
    is_visible = db.Column(db.Boolean, default=True)
    
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
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_main = db.Column(db.Boolean, default=False)

class Employers(db.Model):
    __tablename__ = 'employers'
    employer_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    company_name = db.Column(db.String(255), nullable=False)
    contact_info = db.Column(db.Text, nullable=True)

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
    is_native = db.Column(db.Boolean, default=False)
    employer_id = db.Column(db.Integer, db.ForeignKey('employers.employer_id'), nullable=True)

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

def calculate_matches_background(cv_id, cv_text):
    with app.app_context():
        start_time = time.time()
        print(f"[{start_time}] Starting background matching for CV ID: {cv_id}")
        
        # 1. Check for Cached Embeddings
        job_embeddings, job_ids = load_cached_embeddings()
        
        if job_embeddings is None:
            # Fallback to DB fetch (Slow path)
            print(f"[{time.time()}] Cache miss. Fetching from DB (SLOW)...")
            active_jobs = Job_Descriptions.query.filter_by(active_status=True).all()
            if not active_jobs:
                print("No active jobs found for matching.")
                return 

            job_texts = [clean_text(job.raw_text) for job in active_jobs] # Or use cleaned text if stored
            job_ids = [job.jd_id for job in active_jobs]
            print(f"[{time.time()}] Encoding {len(job_texts)} jobs from DB...")
            job_embeddings = nlp_model.encode(job_texts, convert_to_tensor=True)
        else:
            print(f"[{time.time()}] Using cached embeddings for {len(job_ids)} jobs (FAST).")

        # 2. Encode CV
        cleaned_cv = clean_text(cv_text)
        print(f"[{time.time()}] Encoding User CV...")
        cv_embedding = nlp_model.encode(cleaned_cv, convert_to_tensor=True)

        # 3. Calculate Cosine Similarity
        # cv_embedding is [1, 384], job_embeddings is [N, 384]
        # util.cos_sim returns [1, N]
        scores = util.cos_sim(cv_embedding, job_embeddings)[0]

        # 4. Prepare Scores List
        all_scores = []
        for idx, score in enumerate(scores):
            similarity = float(score)
            all_scores.append({
                'cv_id': cv_id,
                'jd_id': job_ids[idx],
                'similarity_score': similarity
            })
            
        # 5. SORT and KEEP TOP 10 ONLY
        # This drastically reduces DB writes (20k -> 10)
        top_matches = sorted(all_scores, key=lambda x: x['similarity_score'], reverse=True)[:10]

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
        cv_texts = [clean_text(cv.raw_text) for cv in cvs]
        cv_ids = [cv.cv_id for cv in cvs]
        cv_embeddings = nlp_model.encode(cv_texts, convert_to_tensor=True)

        # Encode Jobs
        job_texts = [clean_text(job.raw_text) for job in jobs]
        job_ids = [job.jd_id for job in jobs]
        job_embeddings = nlp_model.encode(job_texts, convert_to_tensor=True)

        # 3. Compute Similarity Matrix
        # 3. Calculate Scores Matrix [N_CVs, M_Jobs]
        # Only if both have items
        if len(cv_embeddings) > 0 and len(job_embeddings) > 0:
            all_scores_matrix = util.cos_sim(cv_embeddings, job_embeddings)
            
            # 4. Save to DB (Optimized)
            new_scores_objects = []
            
            for i, cv in enumerate(cvs):
                # Get scores for this CV against all jobs
                cv_scores = all_scores_matrix[i] # Tensor shape [M_Jobs]
                
                # Pair with Job IDs and Score
                cv_matches = []
                for j, score in enumerate(cv_scores):
                    cv_matches.append({
                        'cv_id': cv.cv_id,
                        'jd_id': job_ids[j],
                        'similarity_score': float(score)
                    })
                
                # Sort and Keep Top 10
                top_10 = sorted(cv_matches, key=lambda x: x['similarity_score'], reverse=True)[:10]
                
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
    
    # 1. Get Job Text/Embedding
    job_text = clean_text(job.raw_text)
    job_embedding = nlp_model.encode(job_text, convert_to_tensor=True)
    
    # 2. Get All Candidates
    cvs = db.session.query(CVs).join(Users, CVs.user_id == Users.user_id).filter(CVs.user_id.isnot(None), Users.is_visible == True).all()
    
    if not cvs:
        flash('No visible candidates found in database.')
        return redirect(url_for('employer_dashboard'))
        
    cv_texts = [clean_text(cv.raw_text) for cv in cvs]
    cv_embeddings = nlp_model.encode(cv_texts, convert_to_tensor=True)
    
    # 3. Compute Similarity
    # job_embedding: [1, 384], cv_embeddings: [N, 384]
    scores = util.cos_sim(job_embedding, cv_embeddings)[0] # [N]
    
    # 4. Find Best Match
    best_score = -1.0
    best_cv_index = -1
    
    for i, score in enumerate(scores):
        if score > best_score:
            best_score = score
            best_cv_index = i
            
    if best_cv_index != -1:
        best_cv = cvs[best_cv_index]
        match_percentage = round(float(best_score) * 100, 1)
        
        # Get User details
        candidate_user = Users.query.get(best_cv.user_id)
        
        # Check if already notified
        has_notified = False
        if Notifications.query.filter_by(user_id=candidate_user.user_id, employer_id=employer_id, job_id=job_id).first():
            has_notified = True

        return render_template('candidate_match.html', 
                               job=job, 
                               candidate=candidate_user, 
                               score=match_percentage,
                               cv=best_cv,
                               has_notified=has_notified)
    else:
        flash('Could not determine a best match.')
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
    user_cvs = CVs.query.filter_by(user_id=current_user.user_id).order_by(CVs.upload_date.desc()).all()

    if request.method == 'POST':
        # Update Profile (Description + Visibility)
        if 'short_description' in request.form:
            current_user.short_description = request.form.get('short_description')
            current_user.is_visible = request.form.get('is_visible') == 'on'
            db.session.commit()
            flash('Profile updated successfully.')
            return redirect(url_for('profile'))

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

# Initialize DB
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
