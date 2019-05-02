import os

from flask import Blueprint, render_template, jsonify
from sh import tail, systemctl

from frontend.auth import login_required
from frontend.persistence import get_static

bp = Blueprint('monitor', __name__, url_prefix='/monitor')

_active_start = 'Active: '
_active_end = ' ('
_since_start = 'since '
_since_end = ';'


@bp.route('/')
@login_required
def index():
    return render_template('monitor/index.html')


_agent_status = systemctl.bake('status', '--output=short', '--no-pager', 'pythia-agent.service')


@bp.route("/status")
@login_required
def status():
    result = _agent_status(_ok_code=[0, 3])
    lines = result.splitlines()
    if len(lines) < 3:
        return jsonify(status="error", message=result)
    line = lines[2]

    try:
        active_str = _parse_enveloping(line, _active_start, _active_end)
        active = active_str == 'active'
        res = {'status': "ok", 'active': active}
        if active:
            res['since'] = _parse_enveloping(line, _since_start, _since_end)
    except _ParsingError:
        return jsonify(status="error", message=result)
    return jsonify(**res)


def _parse_enveloping(line, start_token, end_token):
    line_length = len(line)
    start_offset = len(start_token)
    start = line.find(start_token)
    if start == -1 or (start + start_offset) >= line_length:
        raise _ParsingError()

    line = line[start + start_offset:]
    end = line.find(end_token)
    if end == -1 or end >= line_length:
        raise _ParsingError()
    return line[:end]


class _ParsingError(ValueError):
    pass


@bp.route('/agent/logs/<int:amount>')
@login_required
def agent_logs(amount):
    static = get_static()
    if not os.path.isfile(static.main_log):
        return jsonify(messages=[])

    lines = tail('-n {}'.format(amount), static.main_log).splitlines()
    return jsonify(messages=lines)
