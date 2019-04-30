from flask import Blueprint, request, redirect, url_for, session, g
from werkzeug.security import check_password_hash

from frontend.persistence import get_static

bp = Blueprint('auth', __name__, url_prefix='/auth')


def _validate_login(username, password):
    s = get_static()
    return s.user == username and check_password_hash(s.password, password)


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if _validate_login(username, password):
            session.clear()
            session['valid'] = True
            return redirect(url_for('index'))

    return "Valid login"


@bp.route('/logout')
def logout():
    session.clear()
    return "Logout"


@bp.before_app_request
def validate_session():
    g.valid_session = session.get('valid')
