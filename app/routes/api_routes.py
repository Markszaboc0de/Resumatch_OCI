"""JSON/AJAX endpoints: notifications, suggestions, supersearch, match status,
and the job-sync webhook.

Bug fixes vs the broken refactor:
  * supersearch now imports util / load_cached_embeddings / save_embeddings_cache
    (these were referenced but never imported, raising NameError at runtime).
  * mark_notification_read is POST-only again (state change must not be a GET).
  * the webhook is CSRF-exempt (bearer-token auth), timing-safe, requires the
    secret to be configured, and never leaks tracebacks.
"""
import os
import logging

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
)
from flask_login import login_required, current_user
from sentence_transformers import util  # FIX: was missing in the broken refactor

from app.extensions import db, limiter, csrf
from app.models import Job_Descriptions, Notifications
from app.utils.helpers import clean_text, check_and_deduct_supersearch, is_user_calculating

logger = logging.getLogger('resumatch')

api_bp = Blueprint('api_routes', __name__)


@api_bp.route('/notifications')
@login_required
def notifications():
    user_notifs = Notifications.query.filter_by(user_id=current_user.user_id).order_by(Notifications.timestamp.desc()).all()
    return render_template('notifications.html', notifications=user_notifs)


@api_bp.route('/notifications/mark_read/<int:notification_id>', methods=['POST'])
@login_required
def mark_notification_read(notification_id):
    notif = Notifications.query.get_or_404(notification_id)
    if notif.user_id != current_user.user_id:
        flash('Unauthorized.')
        return redirect(url_for('api_routes.notifications'))

    notif.is_read = True
    db.session.commit()
    return redirect(url_for('api_routes.notifications'))


@api_bp.route('/api/suggestions')
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
        suggestions = [r[0] for r in results if r[0]]
        return jsonify(suggestions)

    return jsonify([])


@api_bp.route('/api/supersearch', methods=['POST'])
@limiter.limit("10 per minute")
@login_required
def supersearch():
    # Heavy imports kept local so importing this blueprint stays light.
    import torch
    from app.services.ai_service import nlp_model, load_cached_embeddings, save_embeddings_cache

    if not check_and_deduct_supersearch(current_user):
        return jsonify({'error': 'You have reached your Supersearch limit for this month.'}), 403

    data = request.get_json()
    if not data or 'ideal_job_description' not in data:
        return jsonify({'error': 'Missing ideal_job_description'}), 400

    location_filter = data.get('location_filter', 'all')
    ideal_desc = data['ideal_job_description']
    if not ideal_desc.strip():
        return jsonify({'error': 'Description cannot be empty'}), 400

    cleaned_desc = clean_text(ideal_desc)
    ideal_embedding = nlp_model.encode(cleaned_desc, convert_to_tensor=True)

    job_embeddings, job_ids = load_cached_embeddings()
    active_jobs = Job_Descriptions.query.filter_by(active_status=True).all()

    def _remaining():
        if current_user.supersearch_limit is None:
            return 'Infinite'
        return current_user.supersearch_limit - current_user.supersearch_used_this_month

    if not active_jobs:
        return jsonify({'matches': [], 'remaining': _remaining()})

    if job_embeddings is None:
        import json
        job_embeddings_list = []
        job_ids = []
        for job in active_jobs:
            if job.parsed_tokens and job.parsed_tokens.startswith('['):
                try:
                    job_embeddings_list.append(json.loads(job.parsed_tokens))
                    job_ids.append(job.jd_id)
                except Exception:
                    logger.debug("Skipping job %s: unparseable parsed_tokens", job.jd_id, exc_info=True)

        if not job_embeddings_list:
            return jsonify({'matches': [], 'remaining': _remaining()}), 200

        job_embeddings = torch.tensor(job_embeddings_list)

        try:
            save_embeddings_cache(job_embeddings, job_ids)
        except Exception:
            logger.warning("Failed to save supersearch embeddings cache", exc_info=True)

    scores = util.cos_sim(ideal_embedding, job_embeddings)[0]

    active_job_ids = {job.jd_id: job for job in active_jobs}

    all_scores = []
    for idx, score in enumerate(scores):
        if job_ids[idx] in active_job_ids:
            job = active_job_ids[job_ids[idx]]
            country = str(job.country).lower() if job.country else ''
            is_hungary = 'hungary' in country or 'magyarország' in country

            if location_filter == 'hungary' and not is_hungary:
                continue
            if location_filter == 'abroad' and is_hungary:
                continue

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

    return jsonify({
        'matches': matches_response,
        'remaining': _remaining()
    })


@api_bp.route('/api/match_status')
@login_required
def match_status():
    is_calculating = is_user_calculating(current_user.user_id)
    return jsonify({'is_calculating': is_calculating})


@api_bp.route('/api/webhook/sync-jobs', methods=['POST'])
@csrf.exempt  # Authenticated via bearer token, not a browser session — CSRF token N/A.
def webhook_sync_jobs():
    import subprocess
    import hmac

    token = request.headers.get('Authorization', '')
    webhook_secret = os.getenv('WEBHOOK_SECRET')
    if not webhook_secret:
        return jsonify({"error": "Server misconfigured: WEBHOOK_SECRET not set"}), 503
    expected_token = f"Bearer {webhook_secret}"

    if not hmac.compare_digest(token, expected_token):
        return jsonify({"error": "Unauthorized"}), 401

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script_path = os.path.join(base_dir, 'import_scraped_jobs.py')

    # Run the import script in the background; it uses fcntl to prevent concurrent runs.
    import sys
    executable = sys.executable or 'python3'
    subprocess.Popen([executable, script_path], cwd=base_dir)

    return jsonify({"message": "Sync triggered successfully. If already running, it will safely exit."}), 202
