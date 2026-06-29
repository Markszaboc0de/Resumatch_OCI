"""Stateless helpers: text extraction/cleaning, quota deduction, HTML sanitizing,
and the cross-worker "is this user's match calc running?" marker files.

All ported verbatim from the original app.py. Functions that need the upload
folder read it from ``current_app.config`` so they work in both the web process
and the Celery worker (which both push an application context).
"""
import os
import re
import html
import time
import json
import logging

import fitz  # PyMuPDF
import nh3
from markupsafe import Markup
from flask import current_app

from app.extensions import db

logger = logging.getLogger('resumatch')

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# A calc marker older than this is treated as stale (worker crashed mid-calc).
CALC_LOCK_TTL = 600  # seconds


# --- HTML sanitizing (stored XSS prevention) ---

def sanitize_html(value):
    """Sanitize stored/scraped HTML (job descriptions) before rendering.

    Registered as the ``clean_html`` Jinja filter in create_app().
    """
    if not value:
        return ''
    return Markup(nh3.clean(value))


# --- File helpers ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    """Robust text cleaning."""
    if not text:
        return ""
    text = html.unescape(text)

    # Remove script and style tags and their contents
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    # Use \w to keep all Unicode letters and numbers, replacing punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def extract_text_from_pdf(filepath):
    try:
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""


def extract_text_from_txt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading TXT: {e}")
        return ""


# --- Quota deduction ---

def check_and_deduct_match(entity):
    import datetime as dt
    now = dt.datetime.utcnow()

    # 1. Reset check
    if entity.match_reset_date is None or now >= entity.match_reset_date:
        entity.matches_used_this_month = 0
        entity.match_reset_date = now + dt.timedelta(days=30)
        db.session.commit()

    # 2. Limit check
    if entity.match_limit is None:  # Infinite
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
    if entity.supersearch_limit is None:  # Infinite
        return True

    if entity.supersearch_used_this_month < entity.supersearch_limit:
        entity.supersearch_used_this_month += 1
        db.session.commit()
        return True

    return False


# --- Cross-worker calc-progress markers ---
# gunicorn workers and the Celery worker don't share memory, so a global set()
# would be per-process and give wrong answers (e.g. /api/match_status). The
# upload folder IS shared across all processes on the host, so marker files are
# correct cross-worker without a DB migration or Redis.

def _calc_lock_path(user_id):
    return os.path.join(current_app.config['UPLOAD_FOLDER'], f'.calc_{user_id}.lock')


def mark_calculating(user_id):
    try:
        with open(_calc_lock_path(user_id), 'w') as f:
            f.write(str(time.time()))
    except OSError as e:
        logger.warning("Could not write calc marker for user %s: %s", user_id, e)


def clear_calculating(user_id):
    try:
        os.remove(_calc_lock_path(user_id))
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Could not clear calc marker for user %s: %s", user_id, e)


def is_user_calculating(user_id):
    try:
        mtime = os.path.getmtime(_calc_lock_path(user_id))
    except OSError:
        return False
    if time.time() - mtime > CALC_LOCK_TTL:
        clear_calculating(user_id)  # stale — worker likely crashed mid-calc
        return False
    return True
