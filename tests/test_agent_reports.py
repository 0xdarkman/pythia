import pytest

ACTIVE_STATUS_RETURN = \
    """● pythia-agent.service - Pythia Frontend Service
       Loaded: loaded (/etc/systemd/system/pythia-agent.service; disabled; vendor pr
       Active: active (running) since Thu 2019-05-02 09:52:25 CEST; 26s ago
     Main PID: 18155 (bash)
        Tasks: 8 (limit: 4583)
       CGroup: /system.slice/pythia-agent.service
               ├─18155 bash /home/pythia/pythia/run_agent.sh /home/pythia/pythia/ins
               └─18160 /home/pythia/pythia/venv/bin/python /home/pythia/pythia/servi
    
    May 02 09:52:29 ubuntu-web-host run_agent.sh[18155]: 2019-05-02 09:52:29.093023:
    May 02 09:52:29 ubuntu-web-host run_agent.sh[18155]: WARNING:tensorflow:From /ho
    May 02 09:52:29 ubuntu-web-host run_agent.sh[18155]: Instructions for updating:
    May 02 09:52:29 ubuntu-web-host run_agent.sh[18155]: Use tf.initializers.varianc
    """
INACTIVE_STATUS_RETURN = \
    """● pythia-agent.service - Pythia Frontend Service
       Loaded: loaded (/etc/systemd/system/pythia-agent.service; disabled; vendor pr
       Active: inactive (dead)
    
    May 02 08:21:17 ubuntu-web-host run_agent.sh[15386]:     return _retry_delayed_i
    May 02 08:21:17 ubuntu-web-host run_agent.sh[15386]:   File "/home/pythia/pythia
    May 02 08:21:17 ubuntu-web-host run_agent.sh[15386]:     "{} no valid data after
    May 02 08:21:17 ubuntu-web-host run_agent.sh[15386]: pythia.core.remote.poloniex
    May 02 08:21:17 ubuntu-web-host run_agent.sh[15386]: Result: [{'date': 0, 'high'
    May 02 08:21:17 ubuntu-web-host run_agent.sh[15386]: 1556776803.9639418: BTC_XEM
    May 02 08:21:21 ubuntu-web-host run_agent.sh[16020]: 2019-05-02 08:21:21.721397:
    May 02 08:21:21 ubuntu-web-host run_agent.sh[16020]: WARNING:tensorflow:From /ho
    May 02 08:21:21 ubuntu-web-host run_agent.sh[16020]: Instructions for updating:
    May 02 08:21:21 ubuntu-web-host run_agent.sh[16020]: Use tf.initializers.varianc
    """
INVALID_STATUS_RETURN = \
    """● pythia-agent.service - Pythia Frontend Service
       Loaded: loaded (/etc/systemd/system/pythia-agent.service; disabled; vendor pr
       Active: active since Thu 2019-05-02 09:52:25 CEST 26s ago
     Main PID: 18155 (bash)
        Tasks: 8 (limit: 4583)
       CGroup: /system.slice/pythia-agent.service
               ├─18155 bash /home/pythia/pythia/run_agent.sh /home/pythia/pythia/ins
               └─18160 /home/pythia/pythia/venv/bin/python /home/pythia/pythia/servi
    
    May 02 09:52:29 ubuntu-web-host run_agent.sh[18155]: 2019-05-02 09:52:29.093023:
    May 02 09:52:29 ubuntu-web-host run_agent.sh[18155]: WARNING:tensorflow:From /ho
    May 02 09:52:29 ubuntu-web-host run_agent.sh[18155]: Instructions for updating:
    May 02 09:52:29 ubuntu-web-host run_agent.sh[18155]: Use tf.initializers.varianc
    """


class AgentStatusStub:
    def __init__(self):
        self.returns = ""

    def __call__(self, *args, **kwargs):
        return self.returns


class LogStub:
    def __init__(self):
        self.lines = []

    def __call__(self, narg, *args, **kwargs):
        assert narg.startswith("-n ")
        n = int(narg[3:])
        return '\n'.join(self.lines[-n:])


class FileStub:
    def __init__(self):
        self.exists = True

    def __call__(self, *args, **kwargs):
        return self.exists


@pytest.fixture
def log(monkeypatch):
    s = LogStub()
    monkeypatch.setattr('frontend.monitor.tail', s)
    return s


@pytest.fixture(autouse=True)
def file(monkeypatch):
    s = FileStub()
    monkeypatch.setattr('os.path.isfile', s)
    return s


@pytest.fixture(autouse=True)
def always_login(auth):
    auth.login()


@pytest.fixture(autouse=True)
def agent_status(monkeypatch):
    stub = AgentStatusStub()
    monkeypatch.setattr("frontend.monitor._agent_status", stub)
    return stub


def test_agent_status_requires_login(client, auth):
    auth.logout()
    assert client.get('/monitor/status').status_code == 302


def test_agent_status_is_parsed_and_returned_as_json(client, agent_status):
    agent_status.returns = ACTIVE_STATUS_RETURN
    assert client.get('/monitor/status').get_json() == {'status': "ok", 'active': True,
                                                        'since': "Thu 2019-05-02 09:52:25 CEST"}


def test_agent_status_is_inactive_when_return_code_is_set_apprioriately(client, agent_status):
    agent_status.returns = INACTIVE_STATUS_RETURN
    assert client.get('/monitor/status').get_json() == {'status': "ok", 'active': False}


def test_agent_status_returns_error_when_process_returns_invalid_stream(client, agent_status):
    agent_status.returns = INVALID_STATUS_RETURN
    assert client.get('/monitor/status').get_json() == {'status': "error",
                                                        'message': INVALID_STATUS_RETURN}


def test_agent_log_reports_nothing_when_log_file_is_not_found(client, file):
    file.exists = False
    assert client.get('/monitor/agent/logs/10').get_json() == {'messages': []}


@pytest.mark.parametrize('amount', (10, 5))
def test_agent_logs_reports_the_last_amount_of_lines_requested(client, log, amount):
    log.lines = make_lines(100)
    assert client.get('/monitor/agent/logs/{}'.format(amount)).get_json() == {'messages': log.lines[-amount:]}


def make_lines(amount):
    return ["Line: {}".format(i) for i in range(amount)]


def test_agent_logs_require_login(client, auth):
    auth.logout()
    assert client.get('/monitor/agent/logs/10').status_code == 302
