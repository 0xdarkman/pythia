import pytest

ACTIVE_STATUS_RETURN = b'\xe2\x97\x8f pythia-agent.service - Pythia Frontend Service\n   Loaded: loaded (/etc/systemd/system/pythia-agent.service; disabled; vendor preset: enabled)\n   Active: active (running) since Wed 2019-05-01 06:20:34 CEST; 2h 23min ago\n Main PID: 7998 (bash)\n    Tasks: 8 (limit: 4583)\n   CGroup: /system.slice/pythia-agent.service\n           \xe2\x94\x9c\xe2\x94\x807998 bash /home/pythia/pythia/run_agent.sh /home/pythia/pythia/instance/fpm_online.json\n           \xe2\x94\x94\xe2\x94\x808005 /home/pythia/pythia/venv/bin/python /home/pythia/pythia/service/fpm_service.py --config /home/pythia/pythia/instance/fpm_online.json\n\nMay 01 06:20:38 ubuntu-web-host run_agent.sh[7998]: 2019-05-01 06:20:38.326579: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA\nMay 01 06:20:38 ubuntu-web-host run_agent.sh[7998]: WARNING:tensorflow:From /home/pythia/pythia/venv/lib/python3.6/site-packages/tflearn/initializations.py:119: UniformUnitScaling.__init__ (from tensorflow.python.ops.init_ops) is deprecated and will be removed in a future version.\nMay 01 06:20:38 ubuntu-web-host run_agent.sh[7998]: Instructions for updating:\nMay 01 06:20:38 ubuntu-web-host run_agent.sh[7998]: Use tf.initializers.variance_scaling instead with distribution=uniform to get equivalent behavior.\n'
INACTIVE_STATUS_RETURN_CODE = 3
INVALID_STATUS_RETURN = b'\xe2\x97\x8f pythia-agent.service - Pythia Frontend Service\n   Loaded: loaded (/etc/systemd/system/pythia-agent.service; disabled; vendor preset: enabled)\n  active (running) Wed 2019-05-01 06:20:34 CEST; 2h 23min ago\n Main PID: 7998 (bash)\n    Tasks: 8 (limit: 4583)\n   CGroup: /system.slice/pythia-agent.service\n           \xe2\x94\x9c\xe2\x94\x807998 bash /home/pythia/pythia/run_agent.sh /home/pythia/pythia/instance/fpm_online.json\n           \xe2\x94\x94\xe2\x94\x808005 /home/pythia/pythia/venv/bin/python /home/pythia/pythia/service/fpm_service.py --config /home/pythia/pythia/instance/fpm_online.json\n\nMay 01 06:20:38 ubuntu-web-host run_agent.sh[7998]: 2019-05-01 06:20:38.326579: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA\nMay 01 06:20:38 ubuntu-web-host run_agent.sh[7998]: WARNING:tensorflow:From /home/pythia/pythia/venv/lib/python3.6/site-packages/tflearn/initializations.py:119: UniformUnitScaling.__init__ (from tensorflow.python.ops.init_ops) is deprecated and will be removed in a future version.\nMay 01 06:20:38 ubuntu-web-host run_agent.sh[7998]: Instructions for updating:\nMay 01 06:20:38 ubuntu-web-host run_agent.sh[7998]: Use tf.initializers.variance_scaling instead with distribution=uniform to get equivalent behavior.\n'


class CheckOutputStub:
    def __init__(self):
        self.returns = b''
        self.returns_code = 0

    def __call__(self, args, **kwargs):
        if self.returns_code > 0:
            from subprocess import CalledProcessError
            raise CalledProcessError(returncode=self.returns_code, cmd=b'test command')
        return self.returns


class CheckOutputSpy(CheckOutputStub):
    def __init__(self):
        super().__init__()
        self.received_args = []

    def __call__(self, args, **kwargs):
        self.received_args = args
        return super().__call__(args, **kwargs)


@pytest.fixture(autouse=True)
def always_login(auth):
    auth.login()


@pytest.fixture(autouse=True)
def process(monkeypatch):
    spy = CheckOutputSpy()
    monkeypatch.setattr("subprocess.check_output", spy)
    return spy


def test_agent_status_invokes_systemctl_correctly(client, process):
    client.get('/monitor/status')
    assert process.received_args == ["systemctl", "status", "--output=short", "--no-pager", "pythia-agent.service"]


def test_agent_status_requires_login(client, auth):
    auth.logout()
    assert client.get('/monitor/status').status_code == 302


def test_agent_status_is_parsed_and_returned_as_json(client, process):
    process.returns = ACTIVE_STATUS_RETURN
    assert client.get('/monitor/status').get_json() == {'status': "ok", 'active': True,
                                                        'since': "Wed 2019-05-01 06:20:34 CEST"}


def test_agent_status_is_inactive_when_return_code_is_set_apprioriately(client, process):
    process.returns_code = INACTIVE_STATUS_RETURN_CODE
    assert client.get('/monitor/status').get_json() == {'status': "ok", 'active': False}


def test_agent_status_returns_error_when_process_returns_invalid_stream(client, process):
    process.returns = INVALID_STATUS_RETURN
    assert client.get('/monitor/status').get_json() == {'status': "error",
                                                        'message': INVALID_STATUS_RETURN.decode('utf-8')}
