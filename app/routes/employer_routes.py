"""Employer portal: session-based auth, job CRUD, candidate matching, notify.

Registered with url_prefix='/employer', so the public URLs are identical to the
original monolith (e.g. /employer/login, /employer/match/<id>).
"""
import logging
import datetime as dt

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import util  # FIX: was missing in the broken refactor

from app.extensions import db, limiter
from app.models import Employers, Job_Descriptions, Precalc_Scores, Notifications, Users, CVs
from app.utils.helpers import clean_text, check_and_deduct_match

logger = logging.getLogger('resumatch')

employer_bp = Blueprint('employer_routes', __name__)


@employer_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute", methods=["POST"])
def employer_login():
    if 'employer_id' in session:
        return redirect(url_for('employer_routes.employer_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        employer = Employers.query.filter_by(username=username).first()

        if employer and check_password_hash(employer.password_hash, password):
            session['employer_id'] = employer.employer_id
            session['employer_name'] = employer.company_name
            flash(f'Welcome, {employer.company_name}!')
            return redirect(url_for('employer_routes.employer_dashboard'))
        else:
            flash('Invalid employer credentials.')
            return redirect(url_for('employer_routes.employer_login'))

    return render_template('employer_login.html')


@employer_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("10 per hour", methods=["POST"])
def employer_register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        company_name = request.form.get('company_name')

        if Employers.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('employer_routes.employer_register'))

        hashed_password = generate_password_hash(password)
        new_employer = Employers(username=username, password_hash=hashed_password, company_name=company_name)
        db.session.add(new_employer)
        db.session.commit()
        flash('Employer account created and pending admin approval. You can log in and post jobs now; '
              'candidate matching unlocks once an admin approves your account.')
        return redirect(url_for('employer_routes.employer_login'))
    return render_template('employer_register.html')


@employer_bp.route('/logout')
def employer_logout():
    session.pop('employer_id', None)
    session.pop('employer_name', None)
    flash('Logged out from Employer Portal.')
    return redirect(url_for('main_routes.landing'))


@employer_bp.route('/dashboard')
def employer_dashboard():
    employer_id = session.get('employer_id')
    if not employer_id:
        flash('Please log in as an employer.')
        return redirect(url_for('employer_routes.employer_login'))

    employer = Employers.query.get(employer_id)
    jobs = Job_Descriptions.query.filter_by(employer_id=employer_id).order_by(Job_Descriptions.jd_id.desc()).all()

    return render_template('employer_dashboard.html', employer=employer, jobs=jobs)


@employer_bp.route('/update_profile', methods=['POST'])
def employer_update_profile():
    employer_id = session.get('employer_id')
    if not employer_id:
        return redirect(url_for('employer_routes.employer_login'))

    employer = Employers.query.get(employer_id)
    if employer:
        employer.company_name = request.form.get('company_name')
        employer.contact_info = request.form.get('contact_info')
        db.session.commit()
        session['employer_name'] = employer.company_name
        flash('Company profile updated.')

    return redirect(url_for('employer_routes.employer_dashboard'))


@employer_bp.route('/create_job', methods=['POST'])
def create_job():
    employer_id = session.get('employer_id')
    if not employer_id:
        return redirect(url_for('employer_routes.employer_login'))

    title = request.form.get('title')
    city = request.form.get('city')
    country = request.form.get('country')
    description = request.form.get('description')
    url = request.form.get('url')
    apply_method = request.form.get('apply_method', 'redirect')

    is_native = (apply_method == 'native')

    if url and not is_native and not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    elif is_native:
        url = None

    employer = Employers.query.get(employer_id)
    company = employer.company_name

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
    except Exception:
        db.session.rollback()
        logger.exception("Error posting job")
        flash('An error occurred while posting the job. Please try again.')

    return redirect(url_for('employer_routes.employer_dashboard'))


@employer_bp.route('/edit_job/<int:job_id>', methods=['POST'])
def edit_job(job_id):
    employer_id = session.get('employer_id')
    if not employer_id:
        return redirect(url_for('employer_routes.employer_login'))

    job = Job_Descriptions.query.get_or_404(job_id)
    if job.employer_id != employer_id:
        flash('Unauthorized.')
        return redirect(url_for('employer_routes.employer_dashboard'))

    job.title = request.form.get('title')
    job.city = request.form.get('city')
    job.country = request.form.get('country')

    description = request.form.get('description')

    if description and description != job.raw_text:
        job.raw_text = description
        job.parsed_tokens = clean_text(description)
        # Clear existing match scores to force recalculation since requirements changed
        Precalc_Scores.query.filter_by(jd_id=job.jd_id).delete()

    apply_method = request.form.get('apply_method', 'redirect')
    job.is_native = (apply_method == 'native')

    url = request.form.get('url')
    if url and not job.is_native and not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    elif job.is_native:
        url = None
    job.url = url

    try:
        db.session.commit()
        flash('Job updated successfully!')
    except Exception:
        db.session.rollback()
        logger.exception("Error updating job")
        flash('An error occurred while updating the job. Please try again.')

    return redirect(url_for('employer_routes.employer_dashboard'))


@employer_bp.route('/toggle_status/<int:job_id>', methods=['POST'])
def toggle_job_status(job_id):
    employer_id = session.get('employer_id')
    if not employer_id:
        return redirect(url_for('employer_routes.employer_login'))

    job = Job_Descriptions.query.get_or_404(job_id)
    if job.employer_id != employer_id:
        flash('Unauthorized.')
        return redirect(url_for('employer_routes.employer_dashboard'))

    job.active_status = not job.active_status
    db.session.commit()
    return redirect(url_for('employer_routes.employer_dashboard'))


@employer_bp.route('/delete_job/<int:job_id>', methods=['POST'])
def delete_job(job_id):
    employer_id = session.get('employer_id')
    if not employer_id:
        return redirect(url_for('employer_routes.employer_login'))

    job = Job_Descriptions.query.get_or_404(job_id)
    if job.employer_id != employer_id:
        flash('Unauthorized.')
        return redirect(url_for('employer_routes.employer_dashboard'))

    Precalc_Scores.query.filter_by(jd_id=job_id).delete()
    Notifications.query.filter_by(job_id=job_id).delete()

    db.session.delete(job)
    db.session.commit()
    flash('Job deleted.')
    return redirect(url_for('employer_routes.employer_dashboard'))


@employer_bp.route('/match/<int:job_id>')
def employer_match_candidate(job_id):
    employer_id = session.get('employer_id')
    if not employer_id:
        return redirect(url_for('employer_routes.employer_login'))

    job = Job_Descriptions.query.get_or_404(job_id)
    if job.employer_id != employer_id:
        flash('Unauthorized.')
        return redirect(url_for('employer_routes.employer_dashboard'))

    employer = Employers.query.get(employer_id)
    # Privacy gate: only approved employers may view candidate CV matches.
    if not employer.is_approved:
        flash('Your employer account is pending admin approval. Candidate matching unlocks once approved.')
        return redirect(url_for('employer_routes.employer_dashboard'))
    if not check_and_deduct_match(employer):
        flash('You have reached your matching limit for this month.')
        return redirect(url_for('employer_routes.employer_dashboard'))

    # Heavy imports kept local so importing this blueprint stays light.
    import json
    import torch
    from app.services.ai_service import nlp_model, extract_match_reasons

    keywords_param = request.args.get('keywords', '')
    filter_keywords = [kw.strip().lower() for kw in keywords_param.split(',') if kw.strip()]

    job_text = clean_text(job.raw_text)
    job_embedding = nlp_model.encode(job_text, convert_to_tensor=True)

    seven_days_ago = dt.datetime.utcnow() - dt.timedelta(days=7)
    all_cvs = db.session.query(CVs).join(Users, CVs.user_id == Users.user_id).filter(
        CVs.user_id.isnot(None),
        Users.is_visible == True,
        Users.last_active_date >= seven_days_ago
    ).all()

    cvs = []
    if filter_keywords:
        for cv in all_cvs:
            cv_raw_lower = cv.raw_text.lower() if cv.raw_text else ""
            if any(kw in cv_raw_lower for kw in filter_keywords):
                cvs.append(cv)
    else:
        cvs = all_cvs

    if not cvs:
        flash('No visible candidates found matching the criteria.')
        return redirect(url_for('employer_routes.employer_dashboard'))

    expected_dim = job_embedding.shape[-1]

    valid_cvs = []
    cv_embeddings_list = []

    for cv in cvs:
        vec = None
        if cv.parsed_tokens and cv.parsed_tokens.startswith('['):
            try:
                parsed_vec = json.loads(cv.parsed_tokens)
                if len(parsed_vec) == expected_dim:
                    vec = parsed_vec
            except Exception:
                logger.debug("Skipping CV %s: unparseable parsed_tokens", cv.cv_id, exc_info=True)

        if vec is None and cv.raw_text:
            vec_tensor = nlp_model.encode(clean_text(cv.raw_text), convert_to_tensor=False)
            vec = vec_tensor.tolist()

        if vec is not None:
            cv_embeddings_list.append(vec)
            valid_cvs.append(cv)

    if not valid_cvs:
        flash('No valid candidates found (no vectors).')
        return redirect(url_for('employer_routes.employer_dashboard'))

    cvs = valid_cvs
    cv_embeddings = torch.tensor(cv_embeddings_list)

    scores = util.cos_sim(job_embedding, cv_embeddings)[0]  # [N]

    all_scores = [(float(score), i) for i, score in enumerate(scores)]
    all_scores.sort(key=lambda x: x[0], reverse=True)

    top_matches = all_scores[:5]

    match_list = []
    if top_matches:
        for score_val, cv_idx in top_matches:
            cv_item = cvs[cv_idx]
            match_percentage = round(score_val * 100, 1)

            candidate_user = Users.query.get(cv_item.user_id)
            reasons = extract_match_reasons(cv_item.extracted_skills, job.extracted_skills)

            has_notified = False
            if Notifications.query.filter_by(user_id=candidate_user.user_id, employer_id=employer_id, job_id=job_id).first():
                has_notified = True

            # Privacy: expose only score, skill basis, public blurb and an opaque id.
            match_list.append({
                'candidate_id': candidate_user.user_id,
                'score': match_percentage,
                'reasons': reasons,
                'has_notified': has_notified,
                'short_description': candidate_user.short_description,
            })

        return render_template('candidate_match.html',
                               job=job,
                               matches=match_list,
                               active_keywords=keywords_param)
    else:
        flash('Could not determine any matches.')
    return redirect(url_for('employer_routes.employer_dashboard'))


@employer_bp.route('/notify/<int:job_id>/<int:candidate_id>', methods=['POST'])
def notify_candidate(job_id, candidate_id):
    employer_id = session.get('employer_id')
    if not employer_id:
        flash('Please log in.')
        return redirect(url_for('employer_routes.employer_login'))

    existing = Notifications.query.filter_by(
        user_id=candidate_id,
        employer_id=employer_id,
        job_id=job_id
    ).first()

    if existing:
        flash('You have already notified this candidate for this job.')
        return redirect(url_for('employer_routes.employer_match_candidate', job_id=job_id))

    employer = Employers.query.get(employer_id)
    job = Job_Descriptions.query.get(job_id)

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

    return redirect(url_for('employer_routes.employer_match_candidate', job_id=job_id))
