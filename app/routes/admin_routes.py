"""Admin portal: session-based auth, dashboard, edit user/employer.

Registered with url_prefix='/admin'.
"""
import datetime as dt

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, session
)
from werkzeug.security import check_password_hash

from app.extensions import db, limiter
from app.models import Users, Employers, Admins

admin_bp = Blueprint('admin_routes', __name__)


@admin_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute", methods=["POST"])
def admin_login():
    if 'admin_id' in session:
        return redirect(url_for('admin_routes.admin_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        admin = Admins.query.filter_by(username=username).first()

        if admin and check_password_hash(admin.password_hash, password):
            session['admin_id'] = admin.admin_id
            session['admin_username'] = admin.username
            flash(f'Welcome Admin {admin.username}!')
            return redirect(url_for('admin_routes.admin_dashboard'))
        else:
            flash('Invalid admin credentials.')
            return redirect(url_for('admin_routes.admin_login'))

    return render_template('admin_login.html')


@admin_bp.route('/logout')
def admin_logout():
    session.pop('admin_id', None)
    session.pop('admin_username', None)
    flash('Logged out from Admin Portal.')
    return redirect(url_for('main_routes.landing'))


@admin_bp.route('/dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        flash('Please log in as an Admin.')
        return redirect(url_for('admin_routes.admin_login'))

    users = Users.query.all()
    employers = Employers.query.all()

    seven_days_ago = dt.datetime.utcnow() - dt.timedelta(days=7)

    return render_template('admin_dashboard.html', users=users, employers=employers, seven_days_ago=seven_days_ago)


@admin_bp.route('/edit_user/<int:user_id>', methods=['POST'])
def admin_edit_user(user_id):
    if 'admin_id' not in session:
        return redirect(url_for('admin_routes.admin_login'))

    user = Users.query.get_or_404(user_id)

    user.is_visible = request.form.get('is_visible') == 'on'
    limit_val = request.form.get('match_limit')

    if not limit_val or limit_val.strip() == '' or limit_val.strip().lower() == 'infinite':
        user.match_limit = None
    else:
        try:
            user.match_limit = int(limit_val)
        except ValueError:
            pass

    try:
        user.matches_used_this_month = int(request.form.get('matches_used_this_month', user.matches_used_this_month))
    except ValueError:
        pass

    ss_limit_val = request.form.get('supersearch_limit')

    if not ss_limit_val or ss_limit_val.strip() == '' or ss_limit_val.strip().lower() == 'infinite':
        user.supersearch_limit = None
    else:
        try:
            user.supersearch_limit = int(ss_limit_val)
        except ValueError:
            pass

    try:
        user.supersearch_used_this_month = int(request.form.get('supersearch_used_this_month', user.supersearch_used_this_month))
    except ValueError:
        pass

    try:
        if 'token_balance' in request.form:
            new_token_balance = int(request.form.get('token_balance'))
            if new_token_balance != user.token_balance:
                diff = new_token_balance - user.token_balance
                user.token_balance = new_token_balance
                from app.models import TokenTransactions
                db.session.add(TokenTransactions(
                    user_id=user.user_id,
                    amount=diff,
                    reason="Admin manual adjustment"
                ))
    except ValueError:
        pass

    db.session.commit()
    flash(f'User {user.username} updated.')
    return redirect(url_for('admin_routes.admin_dashboard'))


@admin_bp.route('/edit_employer/<int:employer_id>', methods=['POST'])
def admin_edit_employer(employer_id):
    if 'admin_id' not in session:
        return redirect(url_for('admin_routes.admin_login'))

    employer = Employers.query.get_or_404(employer_id)

    employer.is_approved = request.form.get('is_approved') == 'on'

    limit_val = request.form.get('match_limit')

    if not limit_val or limit_val.strip() == '' or limit_val.strip().lower() == 'infinite':
        employer.match_limit = None
    else:
        try:
            employer.match_limit = int(limit_val)
        except ValueError:
            pass

    try:
        employer.matches_used_this_month = int(request.form.get('matches_used_this_month', employer.matches_used_this_month))
    except ValueError:
        pass

    db.session.commit()
    flash(f'Employer {employer.company_name} updated.')
    return redirect(url_for('admin_routes.admin_dashboard'))
