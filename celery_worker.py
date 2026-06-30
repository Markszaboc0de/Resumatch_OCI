"""Celery worker entrypoint.

Importing the app package runs create_app(), which configures Celery's broker /
backend and registers the ContextTask (so every task runs inside a Flask app
context). The task modules are imported during create_app() (the route modules
import app.services.ai_service), so the @celery.task functions are registered by
the time this module finishes importing.

Run with:
    celery -A celery_worker.celery worker --loglevel=info
"""
from app import app  # noqa: F401  (runs create_app(), configures celery)
from app.extensions import celery  # noqa: F401  (the configured Celery instance)

import app.services.ai_service  # noqa: F401,E402

from celery.signals import worker_process_init
from app.extensions import db

@worker_process_init.connect
def dispose_sqlalchemy_pool(**kwargs):
    """Dispose the connection pool in each Celery worker process so they don't share the parent's socket."""
    with app.app_context():
        db.engine.dispose()
