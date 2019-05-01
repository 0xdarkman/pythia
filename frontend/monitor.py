import subprocess

from flask import Blueprint, render_template, jsonify

from frontend.auth import login_required

bp = Blueprint('monitor', __name__, url_prefix='/monitor')

_active_start = 'Active: '
_active_end = ' (running)'
_since_start = 'since '
_since_end = ';'


@bp.route('/')
@login_required
def index():
    return render_template('monitor/index.html')


@bp.route("/status")
@login_required
def status():
    try:
        result = subprocess.check_output(
            ["systemctl", "status", "--output=short", "--no-pager", "pythia-agent.service"])
        result = result.decode('utf-8')
    except subprocess.CalledProcessError as error:
        if error.returncode == 3:
            return jsonify(status="ok", active=False)
        raise

    lines = result.splitlines()
    if len(lines) < 3:
        return jsonify(status="error", message=result)
    line = lines[2]

    try:
        active_str = _parse_enveloping(line, _active_start, _active_end)
        since_str = _parse_enveloping(line, _since_start, _since_end)
    except _ParsingError:
        return jsonify(status="error", message=result)
    return jsonify(status="ok", active=active_str == 'active', since=since_str)


def _parse_enveloping(line, start_token, end_token):
    start_offset = len(start_token)
    start = line.find(start_token)
    end = line.find(end_token)
    line_length = len(line)
    if start == -1 or end == -1 or (start + start_offset) >= line_length or end >= line_length:
        raise _ParsingError()
    return line[line.find(start_token) + len(start_token):line.find(end_token)]


class _ParsingError(ValueError):
    pass
