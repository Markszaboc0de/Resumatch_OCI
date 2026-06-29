"""SQLAlchemy models.

Ported verbatim from the original monolithic app.py so the database schema and
all application behaviour (token economy, employer approval gate, surveys) are
preserved exactly.
"""
from datetime import datetime
from flask_login import UserMixin

from .extensions import db


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

    token_balance = db.Column(db.Integer, default=0, nullable=False)
    surveys_filled_count = db.Column(db.Integer, default=0, nullable=False)

    def get_id(self):
        return str(self.user_id)

    # Relationship to Notifications
    notifications = db.relationship('Notifications', backref='user', lazy=True)


class CVs(db.Model):
    __tablename__ = 'cvs'
    cv_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)  # Integer as per DB schema
    filename = db.Column(db.String(255), nullable=True)  # Store the filename
    file_path = db.Column(db.String(512), nullable=True)  # Store full path
    raw_text = db.Column(db.Text, nullable=False)
    parsed_tokens = db.Column(db.Text, nullable=True)  # JSON or specific format if needed
    extracted_skills = db.Column(db.Text, nullable=True)  # JSON list of extracted NER skills
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_main = db.Column(db.Boolean, default=False)


class Employers(db.Model):
    __tablename__ = 'employers'
    employer_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    company_name = db.Column(db.String(255), nullable=False)
    contact_info = db.Column(db.Text, nullable=True)
    # New self-registered employers start unapproved; an admin must approve before
    # they can view candidate matches (privacy gate).
    is_approved = db.Column(db.Boolean, default=False)

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
    extracted_skills = db.Column(db.Text, nullable=True)  # JSON list of extracted NER skills
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


class SurveyCompletions(db.Model):
    __tablename__ = 'survey_completions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    survey_id = db.Column(db.Integer, db.ForeignKey('surveys.survey_id'), nullable=False)
    started_at = db.Column(db.DateTime, nullable=False)
    claimed_at = db.Column(db.DateTime, nullable=True)
    tokens_awarded = db.Column(db.Integer, nullable=True)

    __table_args__ = (db.UniqueConstraint('user_id', 'survey_id'),)


class TokenTransactions(db.Model):
    __tablename__ = 'token_transactions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    amount = db.Column(db.Integer, nullable=False)   # positive = earn, negative = spend
    reason = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
