"""Job-seeker routes: auth, dashboard, profile/CV upload, matches, surveys, tokens."""
import os
import json
import time
import logging
from datetime import datetime

import fitz  # PyMuPDF
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, jsonify
)
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from app.extensions import db, limiter
from app.models import (
    Users, CVs, Job_Descriptions, Precalc_Scores, Surveys, SurveyCompletions, TokenTransactions
)
from app.utils.helpers import (
    allowed_file, clean_text, check_and_deduct_match, check_and_deduct_supersearch,
    mark_calculating,
)
from app.services.ai_service import ensure_jit_matches, run_upload_jit_full_task

logger = logging.getLogger('resumatch')

user_bp = Blueprint('user_routes', __name__)


@user_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("10 per hour", methods=["POST"])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not email or not username or not password or not confirm_password:
            flash('All fields are required.')
            return redirect(url_for('user_routes.register'))

        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('user_routes.register'))

        if Users.query.filter_by(email=email).first() or Users.query.filter_by(username=username).first():
            flash('That email or username is not available.')
            return redirect(url_for('user_routes.register'))

        hashed_password = generate_password_hash(password)
        new_user = Users(email=email, username=username, password_hash=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('user_routes.dashboard'))
        except Exception:
            db.session.rollback()
            flash('An error occurred during registration.')
            logger.exception("Registration error")
            return redirect(url_for('user_routes.register'))

    return render_template('register.html')


@user_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute", methods=["POST"])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = Users.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)

            # Just-In-Time (JIT) Match Calculation
            ensure_jit_matches(user)

            user.last_active_date = datetime.utcnow()
            db.session.commit()

            return redirect(url_for('user_routes.dashboard'))
        else:
            flash('Invalid email or password.')
            return redirect(url_for('user_routes.login'))

    return render_template('login.html')


@user_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main_routes.landing'))


@user_bp.route('/dashboard')
@login_required
def dashboard():
    ensure_jit_matches(current_user)

    page = request.args.get('page', 1, type=int)
    per_page = 20

    search_query = request.args.get('search', '')
    country_filter = request.args.get('country', '')
    city_filter = request.args.get('city', '')
    company_filter = request.args.get('company', '')

    query = Job_Descriptions.query.filter_by(active_status=True)

    if search_query:
        query = query.filter(
            Job_Descriptions.raw_text.ilike(f'%{search_query}%') |
            Job_Descriptions.title.ilike(f'%{search_query}%')
        )
    if country_filter:
        query = query.filter(Job_Descriptions.country.ilike(f'%{country_filter}%'))
    if city_filter:
        query = query.filter(Job_Descriptions.city.ilike(f'%{city_filter}%'))
    if company_filter:
        query = query.filter(Job_Descriptions.company.ilike(f'%{company_filter}%'))

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        'dashboard.html', jobs=pagination.items, page=page, total_pages=pagination.pages,
        has_next=pagination.has_next, has_prev=pagination.has_prev,
        search_query=search_query, country_filter=country_filter,
        city_filter=city_filter, company_filter=company_filter
    )


@user_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    ensure_jit_matches(current_user)

    user_cvs = CVs.query.filter_by(user_id=current_user.user_id).order_by(CVs.upload_date.desc()).all()

    if request.method == 'POST':
        # Update Profile (Description + Visibility)
        if 'short_description' in request.form:
            current_user.short_description = request.form.get('short_description')
            current_user.is_visible = request.form.get('is_visible') == 'on'
            current_user.last_active_date = datetime.utcnow()
            db.session.commit()
            flash('Profile updated and active status renewed successfully.')
            return redirect(url_for('user_routes.profile'))

        # Check CV Limit
        if len(user_cvs) >= 5:
            flash('You have reached the limit of 5 CVs. Please delete one to upload a new one.')
            return redirect(url_for('user_routes.profile'))

        if 'resume' not in request.files:
            flash('No file part')
            return redirect(url_for('user_routes.profile'))

        file = request.files['resume']

        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('user_routes.profile'))

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > 5 * 1024 * 1024:
            flash('CV file size exceeds the 5MB limit.')
            return redirect(url_for('user_routes.profile'))

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                unique_filename = f"{current_user.user_id}_{int(time.time())}_{filename}"

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

                cleaned_text = clean_text(extracted_text)

                new_cv = CVs(
                    user_id=current_user.user_id,
                    filename=unique_filename,
                    file_path=None,
                    raw_text=extracted_text,
                    parsed_tokens=None,   # Processing State
                    extracted_skills=None
                )
                db.session.add(new_cv)
                db.session.commit()

                # Trigger background AI Analysis & Match Calculation via Celery
                mark_calculating(current_user.user_id)
                run_upload_jit_full_task.delay(current_user.user_id, new_cv.cv_id, cleaned_text)

                flash('Resume uploaded successfully! AI analysis in progress... You can browse the site while you wait.')
                return redirect(url_for('user_routes.profile'))
            except Exception:
                db.session.rollback()
                logger.exception("Error during upload")
                flash('An error occurred during upload. Please check the file and try again.')
                return redirect(url_for('user_routes.profile'))

    import datetime as dt
    seven_days_ago = dt.datetime.utcnow() - dt.timedelta(days=7)
    is_expired = False
    if current_user.last_active_date and current_user.last_active_date < seven_days_ago:
        is_expired = True

    user_survey = Surveys.query.filter_by(user_id=current_user.user_id).first()

    survey_rank = None
    if user_survey:
        rank = (
            db.session.query(db.func.count())
            .select_from(Surveys)
            .join(Users, Surveys.user_id == Users.user_id)
            .filter(Users.surveys_filled_count > current_user.surveys_filled_count)
            .scalar()
        )
        survey_rank = rank + 1

    return render_template('profile.html', user=current_user, cvs=user_cvs, is_expired=is_expired,
                           user_survey=user_survey, survey_rank=survey_rank)


@user_bp.route('/delete_cv/<int:cv_id>', methods=['POST'])
@login_required
def delete_cv(cv_id):
    cv = CVs.query.get_or_404(cv_id)

    if cv.user_id != current_user.user_id:
        flash('Unauthorized action.')
        return redirect(url_for('user_routes.profile'))

    try:
        if cv.file_path and os.path.exists(cv.file_path):
            os.remove(cv.file_path)

        Precalc_Scores.query.filter_by(cv_id=cv_id).delete()

        db.session.delete(cv)
        db.session.commit()
        flash('CV deleted successfully.')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting CV: {e}')

    return redirect(url_for('user_routes.profile'))


@user_bp.route('/set_main_cv/<int:cv_id>', methods=['POST'])
@login_required
def set_main_cv(cv_id):
    cv = CVs.query.get_or_404(cv_id)
    if cv.user_id != current_user.user_id:
        flash('Unauthorized action.')
        return redirect(url_for('user_routes.profile'))

    try:
        CVs.query.filter_by(user_id=current_user.user_id).update({'is_main': False})
        cv.is_main = True
        db.session.commit()
        flash('Main resume updated successfully.')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating main resume: {e}')

    return redirect(url_for('user_routes.profile'))


def _apply_location_filter(query, location_filter):
    if location_filter == 'hungary':
        from sqlalchemy import or_
        return query.filter(
            or_(
                Job_Descriptions.country.ilike('%hungary%'),
                Job_Descriptions.country.ilike('%magyarország%')
            )
        )
    elif location_filter == 'abroad':
        from sqlalchemy import or_, not_
        return query.filter(
            or_(
                Job_Descriptions.country.is_(None),
                not_(
                    or_(
                        Job_Descriptions.country.ilike('%hungary%'),
                        Job_Descriptions.country.ilike('%magyarország%')
                    )
                )
            )
        )
    return query


@user_bp.route('/dream_job')
@login_required
def dream_job():
    ensure_jit_matches(current_user)

    if 'location_filter' not in request.args:
        if not check_and_deduct_match(current_user):
            flash("You have reached your matching limit for this month.")
            return redirect(url_for('user_routes.profile'))
    else:
        if current_user.match_limit is not None and current_user.matches_used_this_month >= current_user.match_limit:
            flash("You have reached your matching limit for this month.")
            return redirect(url_for('user_routes.profile'))

    user_cv = CVs.query.filter_by(user_id=current_user.user_id, is_main=True).first()
    if not user_cv:
        user_cv = CVs.query.filter_by(user_id=current_user.user_id).order_by(CVs.upload_date.desc()).first()

    if not user_cv:
        flash("Please upload a resume first to see your dream job matches.")
        return redirect(url_for('user_routes.profile'))

    if not user_cv.parsed_tokens:
        flash("Your resume is currently being analyzed by our AI in the background. Please check back in a minute!")
        return redirect(url_for('user_routes.profile'))

    query = db.session.query(Precalc_Scores, Job_Descriptions).\
        join(Job_Descriptions, Precalc_Scores.jd_id == Job_Descriptions.jd_id).\
        filter(Precalc_Scores.cv_id == user_cv.cv_id)

    location_filter = request.args.get('location_filter', 'all').lower()
    query = _apply_location_filter(query, location_filter)

    results = query.order_by(Precalc_Scores.similarity_score.desc()).limit(3).all()

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

    processing_jobs = db.session.query(Job_Descriptions.jd_id).filter(
        Job_Descriptions.active_status == True,
        Job_Descriptions.parsed_tokens == None
    ).first()
    better_matches_incoming = processing_jobs is not None

    return render_template('dream_jobs.html', matches=matches,
                           better_matches_incoming=better_matches_incoming,
                           location_filter=location_filter)


@user_bp.route('/matches/<int:cv_id>')
@login_required
def view_matches(cv_id):
    cv = CVs.query.get_or_404(cv_id)
    if cv.user_id != current_user.user_id:
        flash('Unauthorized action.')
        return redirect(url_for('user_routes.profile'))

    if 'location_filter' not in request.args:
        if not check_and_deduct_match(current_user):
            flash('You have reached your matching limit for this month.')
            return redirect(url_for('user_routes.profile'))
    else:
        if current_user.match_limit is not None and current_user.matches_used_this_month >= current_user.match_limit:
            flash("You have reached your matching limit for this month.")
            return redirect(url_for('user_routes.profile'))

    query = db.session.query(Precalc_Scores, Job_Descriptions).\
        join(Job_Descriptions, Precalc_Scores.jd_id == Job_Descriptions.jd_id).\
        filter(Precalc_Scores.cv_id == cv_id)

    location_filter = request.args.get('location_filter', 'all').lower()
    query = _apply_location_filter(query, location_filter)

    results = query.order_by(Precalc_Scores.similarity_score.desc()).limit(50).all()

    # Import here to avoid a heavy import at module load (pulls in torch via service)
    from app.services.ai_service import extract_match_reasons

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

    return render_template('job_seeker.html', matches=matches, cv_id=cv_id, location_filter=location_filter)


# --- SURVEY ROUTES ---

@user_bp.route('/survey')
def survey_board():
    surveys = (
        db.session.query(Surveys)
        .join(Users, Surveys.user_id == Users.user_id)
        .order_by(Users.surveys_filled_count.desc(), Surveys.created_at.desc())
        .all()
    )
    completed_ids = set()
    if current_user.is_authenticated:
        completed_ids = {
            sc.survey_id for sc in SurveyCompletions.query.filter_by(
                user_id=current_user.user_id
            ).filter(SurveyCompletions.claimed_at.isnot(None)).all()
        }
    return render_template('survey.html', surveys=surveys, completed_ids=completed_ids)


@user_bp.route('/profile/survey', methods=['POST'])
@login_required
def save_survey():
    survey_name = request.form.get('survey_name', '').strip()
    survey_description = request.form.get('survey_description', '').strip()
    survey_url = request.form.get('survey_url', '').strip()
    estimated_minutes = request.form.get('estimated_minutes', '').strip()

    if not survey_name or not survey_description or not survey_url or not estimated_minutes:
        flash('All survey fields are required.')
        return redirect(url_for('user_routes.profile'))

    if not survey_url.startswith(('http://', 'https://')):
        survey_url = 'https://' + survey_url

    try:
        minutes = int(estimated_minutes)
        if minutes < 1 or minutes > 120:
            flash('Estimated time must be between 1 and 120 minutes.')
            return redirect(url_for('user_routes.profile'))
    except ValueError:
        flash('Estimated time must be a number.')
        return redirect(url_for('user_routes.profile'))

    if current_user.token_balance < 1:
        flash('You need at least 1 token to publish or update a survey. Earn tokens by filling surveys on the Survey Board.')
        return redirect(url_for('user_routes.profile'))

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

    current_user.token_balance -= 1
    db.session.add(TokenTransactions(
        user_id=current_user.user_id,
        amount=-1,
        reason='Published/updated survey'
    ))
    db.session.commit()
    flash('Survey saved successfully!')
    return redirect(url_for('user_routes.profile'))


@user_bp.route('/profile/survey/delete', methods=['POST'])
@login_required
def delete_survey():
    survey = Surveys.query.filter_by(user_id=current_user.user_id).first()
    if survey:
        db.session.delete(survey)
        db.session.commit()
        flash('Survey removed.')
    return redirect(url_for('user_routes.profile'))


@user_bp.route('/survey/<int:survey_id>/start')
@login_required
def survey_start(survey_id):
    survey = Surveys.query.get_or_404(survey_id)
    existing = SurveyCompletions.query.filter_by(
        user_id=current_user.user_id, survey_id=survey_id
    ).first()
    if existing is None:
        db.session.add(SurveyCompletions(
            user_id=current_user.user_id,
            survey_id=survey_id,
            started_at=datetime.utcnow()
        ))
        db.session.commit()
    elif existing.claimed_at is None:
        existing.started_at = datetime.utcnow()
        db.session.commit()
    return redirect(survey.survey_url)


@user_bp.route('/survey/<int:survey_id>/claim', methods=['POST'])
@login_required
def survey_claim(survey_id):
    survey = Surveys.query.get_or_404(survey_id)

    if survey.user_id == current_user.user_id:
        return jsonify({'error': 'Cannot claim your own survey.'}), 400

    completion = SurveyCompletions.query.filter_by(
        user_id=current_user.user_id, survey_id=survey_id
    ).first()

    if completion is None:
        return jsonify({'error': 'Survey not started. Click the survey link first.'}), 400

    if completion.claimed_at is not None:
        return jsonify({'error': 'Already claimed.'}), 400

    elapsed = (datetime.utcnow() - completion.started_at).total_seconds()
    required = survey.estimated_minutes * 60 * 0.5
    if elapsed < required:
        remaining_secs = int(required - elapsed)
        return jsonify({'error': f'Too soon — please wait {remaining_secs} more second(s).'}), 400

    tokens = max(1, survey.estimated_minutes // 5)
    current_user.token_balance += tokens
    current_user.surveys_filled_count += 1
    completion.claimed_at = datetime.utcnow()
    completion.tokens_awarded = tokens
    db.session.add(TokenTransactions(
        user_id=current_user.user_id,
        amount=tokens,
        reason=f'Filled survey: {survey.survey_name[:200]}'
    ))
    db.session.commit()

    return jsonify({'tokens_awarded': tokens, 'new_balance': current_user.token_balance})


@user_bp.route('/tokens/spend', methods=['POST'])
@login_required
def tokens_spend():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request.'}), 400

    reward = data.get('reward')
    if reward == 'match':
        if current_user.match_limit is None:
            return jsonify({'error': 'Your match limit is already unlimited.'}), 400
        cost = 2
        reason = 'Purchased 1 extra job match'
    elif reward == 'supersearch':
        if current_user.supersearch_limit is None:
            return jsonify({'error': 'Your supersearch limit is already unlimited.'}), 400
        cost = 3
        reason = 'Purchased 1 extra supersearch'
    else:
        return jsonify({'error': 'Unknown reward type.'}), 400

    if current_user.token_balance < cost:
        return jsonify({'error': f'Not enough tokens. Need {cost}, have {current_user.token_balance}.'}), 400

    current_user.token_balance -= cost
    if reward == 'match':
        current_user.match_limit += 1
        new_limit = current_user.match_limit
    else:
        current_user.supersearch_limit += 1
        new_limit = current_user.supersearch_limit

    db.session.add(TokenTransactions(
        user_id=current_user.user_id,
        amount=-cost,
        reason=reason
    ))
    db.session.commit()

    return jsonify({'new_balance': current_user.token_balance, 'new_limit': new_limit})


@user_bp.route('/profile/tokens')
@login_required
def profile_tokens():
    transactions = (
        TokenTransactions.query
        .filter_by(user_id=current_user.user_id)
        .order_by(TokenTransactions.created_at.desc())
        .limit(50)
        .all()
    )
    return render_template('tokens.html', transactions=transactions)
