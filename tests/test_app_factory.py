import pytest

from frontend import create_app


@pytest.fixture(autouse=True)
def from_json(monkeypatch):
    class Recorder:
        file = None
        silent = False

    def fake_from_json(cfg, file, silent=False):
        Recorder.file = file
        Recorder.silent = silent

    monkeypatch.setattr('flask.config.Config.from_json', fake_from_json)
    return Recorder


@pytest.fixture(autouse=True)
def makedirs(monkeypatch):
    class Recorder:
        path = None
        exist_ok = False

    def fake_makedirs(path, **kwargs):
        Recorder.path = path
        Recorder.exist_ok = kwargs.get('exist_ok', False)

    monkeypatch.setattr('os.makedirs', fake_makedirs)
    return Recorder


def test_creating_non_test_env_by_default():
    assert not create_app().testing


def test_not_using_dev_secret_by_default():
    assert 'dev' != create_app().config['SECRET_KEY']


def test_load_config_from_json_by_default(from_json):
    create_app()
    assert from_json.file == "config.json" and from_json.silent


def test_creating_a_test_environment():
    app = create_app({'TESTING': True})
    assert app.testing and app.config['SECRET_KEY'] == 'dev'


def test_create_instance_path(makedirs):
    app = create_app()
    assert makedirs.path == app.instance_path and makedirs.exist_ok


def test_default_creation_uses_instance_path_as_data_dir():
    app = create_app()
    assert app.config['DATA_DIR'] == app.instance_path
