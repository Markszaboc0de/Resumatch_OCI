"""Flask extensions, instantiated without an app.

Each extension is bound to the application inside ``create_app`` via ``init_app``.
Keeping them here (separate from the factory) avoids circular imports: models,
services and routes import the singletons from this module without importing the
factory.
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_babel import Babel
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from celery import Celery

db = SQLAlchemy()
login_manager = LoginManager()
babel = Babel()

# CSRF protection for all state-changing form/AJAX POSTs (see GDPR/security audit).
csrf = CSRFProtect()

# Rate limiting (brute-force / abuse protection). No global default so polling
# endpoints (e.g. /api/match_status, /api/suggestions) are unaffected; sensitive
# routes opt in with @limiter.limit(...). For multiple gunicorn workers, set
# RATELIMIT_STORAGE_URI to a shared store (e.g. redis://...).
limiter = Limiter(key_func=get_remote_address)

# Background task queue. Broker/backend are configured in create_app() from env.
celery = Celery('resumatch')
