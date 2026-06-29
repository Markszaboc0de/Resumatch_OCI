"""Public pages: landing, language switch, job board listings, job detail, map."""
import os
import json
import logging

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, session, current_app
)
from flask_login import current_user

from app.extensions import db
from app.models import Job_Descriptions

logger = logging.getLogger('resumatch')

main_bp = Blueprint('main_routes', __name__)


@main_bp.route('/set_language/<lang>')
def set_language(lang):
    if lang in ['en', 'hu']:
        session['lang'] = lang
    return redirect(request.referrer or url_for('main_routes.landing'))


@main_bp.route('/')
def landing():
    return render_template('landing.html')


@main_bp.route('/employer')
def employer():
    return "Employer specific portal - Coming Soon"


@main_bp.route('/job/<int:jd_id>')
def job_detail(jd_id):
    job = Job_Descriptions.query.get_or_404(jd_id)
    if not job.active_status:
        flash("This job is no longer active.")
        if current_user.is_authenticated:
            return redirect(url_for('user_routes.dashboard'))
        return redirect(url_for('main_routes.listings'))
    return render_template('job_detail.html', job=job)


def _job_board_query():
    """Shared filter/pagination logic for /dashboard and /listings."""
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
    return pagination, page, search_query, country_filter, city_filter, company_filter


@main_bp.route('/listings')
def listings():
    pagination, page, search_query, country_filter, city_filter, company_filter = _job_board_query()
    return render_template(
        'listings.html', jobs=pagination.items, page=page, total_pages=pagination.pages,
        has_next=pagination.has_next, has_prev=pagination.has_prev,
        search_query=search_query, country_filter=country_filter,
        city_filter=city_filter, company_filter=company_filter
    )


@main_bp.route('/map')
def job_map():
    if 'employer_id' in session:
        return redirect(url_for('employer_routes.employer_dashboard'))

    jobs = db.session.query(
        Job_Descriptions.jd_id,
        Job_Descriptions.title,
        Job_Descriptions.company,
        Job_Descriptions.url,
        Job_Descriptions.city,
        Job_Descriptions.country
    ).filter(Job_Descriptions.active_status == True).all()

    cache = {}
    cache_path = os.path.join(current_app.root_path, '..', 'location_cache.json')
    cache_path = os.path.abspath(cache_path)
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            try:
                cache = json.load(f)
            except Exception:
                logger.warning("Failed to parse location_cache.json", exc_info=True)

    map_data = {}
    total_mapped = 0

    for job in jobs:
        if not job.city:
            continue

        key = f"{job.city},{job.country}" if job.country else f"{job.city}"
        coords = cache.get(key)

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

    return render_template('map.html', map_data=map_data, total_mapped_jobs=total_mapped)
