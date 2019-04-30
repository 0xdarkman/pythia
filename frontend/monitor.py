from flask import Blueprint, render_template

from frontend.auth import login_required

bp = Blueprint('monitor', __name__, url_prefix='/monitor')


@bp.route('/')
@login_required
def index():
    return render_template('monitor/index.html')
