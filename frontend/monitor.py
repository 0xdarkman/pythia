from flask import Blueprint

bp = Blueprint('monitor', __name__, url_prefix='/monitor')


@bp.route('/')
def index():
    return "Nothing"
