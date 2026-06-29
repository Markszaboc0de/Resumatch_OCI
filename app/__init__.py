"""Application factory.

Builds the Flask app, binds the extensions, registers the blueprints and wires
Celery. All the security hardening from the original monolith is preserved:
fail-closed secrets, hardened session cookies, CSRF protection, rate limiting and
the nh3 HTML sanitizer.
"""
import os
import logging

from flask import Flask, session
from dotenv import load_dotenv

from .extensions import db, login_manager, babel, csrf, limiter, celery

# Project root (the directory that used to contain the monolithic app.py).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env (critical for Gunicorn / Celery workers).
load_dotenv(os.path.join(BASE_DIR, '.env'))

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)
logger = logging.getLogger('resumatch')


def get_locale():
    return session.get('lang', 'en')


def init_celery(app):
    """Bind Celery to the Flask app config and push an app context per task."""
    celery.conf.broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    celery.conf.result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, 'templates'),
        static_folder=os.path.join(BASE_DIR, 'static'),
    )

    # --- Secrets: fail closed, no shipped defaults ---
    secret_key = os.getenv('SECRET_KEY')
    if not secret_key:
        raise RuntimeError(
            "SECRET_KEY is not set. Define it in .env. "
            "Refusing to start with an insecure default (would allow session forgery)."
        )
    app.config['SECRET_KEY'] = secret_key

    upload_folder_name = os.getenv('UPLOAD_FOLDER', 'uploads')
    app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, upload_folder_name)
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

    # --- Session cookie hardening ---
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    # Secure cookies require HTTPS. On by default; set SESSION_COOKIE_SECURE=False for local HTTP.
    app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'True').lower() != 'false'

    # --- Database: no hardcoded credential fallback ---
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Define it in .env. "
            "Refusing to start without explicit DB credentials."
        )
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # --- Babel ---
    app.config['BABEL_DEFAULT_LOCALE'] = 'en'
    app.config['BABEL_TRANSLATION_DIRECTORIES'] = os.path.join(BASE_DIR, 'translations')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # --- Initialise extensions ---
    db.init_app(app)

    login_manager.init_app(app)
    login_manager.login_view = 'user_routes.login'

    babel.init_app(app, locale_selector=get_locale)

    csrf.init_app(app)

    # For multiple gunicorn workers, point RATELIMIT_STORAGE_URI at a shared store.
    limiter.init_app(app)

    init_celery(app)

    from app.models import Users, Notifications

    @login_manager.user_loader
    def load_user(user_id):
        return Users.query.get(int(user_id))

    # nh3 HTML sanitizer (stored XSS prevention) used by templates as |clean_html
    from app.utils.helpers import sanitize_html
    app.add_template_filter(sanitize_html, 'clean_html')

    # --- Register blueprints (URLs match the original monolith) ---
    from .routes.main_routes import main_bp
    from .routes.user_routes import user_bp
    from .routes.employer_routes import employer_bp
    from .routes.admin_routes import admin_bp
    from .routes.api_routes import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(employer_bp, url_prefix='/employer')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(api_bp)

    from flask_login import current_user

    @app.context_processor
    def inject_notifications():
        if current_user.is_authenticated:
            unread_count = Notifications.query.filter_by(user_id=current_user.user_id, is_read=False).count()
            return dict(unread_count=unread_count)
        return dict(unread_count=0)

    with app.app_context():
        db.create_all()

    return app


app = create_app()

# --- Backward-compatibility re-exports ---
# Operational/maintenance scripts (recalculate.py, import_scraped_jobs.py,
# geocoder.py, export_jobs_csv.py, gunicorn.conf.py, Migrations/, Tests/) import
# names directly from the top-level ``app`` package. Re-export them here so those
# scripts keep working unchanged after the refactor.
from app.models import (  # noqa: E402
    Users, CVs, Employers, Job_Descriptions, Precalc_Scores,
    Notifications, Admins, Surveys, SurveyCompletions, TokenTransactions,
)
from app.utils.helpers import (  # noqa: E402
    allowed_file, clean_text, extract_text_from_pdf, extract_text_from_txt,
    check_and_deduct_match, check_and_deduct_supersearch,
    mark_calculating, clear_calculating, is_user_calculating,
)
from app.services.ai_service import (  # noqa: E402
    nlp_model, ner_model, extract_skills_from_text, extract_match_reasons,
    calculate_matches_background, refresh_all_matches, ensure_jit_matches,
)
