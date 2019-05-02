import json
import os
import shutil
import tempfile

import pytest
from werkzeug.security import generate_password_hash

from frontend import create_app

VALID_PASSWORD = "secret"
VALID_NAME = "name"


def make_tmp():
    return tempfile.mkdtemp()


@pytest.fixture
def username():
    return VALID_NAME


@pytest.fixture
def password():
    return VALID_PASSWORD


@pytest.fixture
def app(username, password):
    tmp = make_tmp()
    with open(os.path.join(tmp, 'static.json'), 'w') as f:
        json.dump({'user': username, 'password': generate_password_hash(password), 'main_log': "main.log"}, f)
    yield create_app({'TESTING': True, 'DATA_DIR': tmp})
    shutil.rmtree(tmp)


@pytest.fixture
def client(app):
    return app.test_client()


class AuthActions:
    def __init__(self, client):
        self._client = client

    def login(self, username=VALID_NAME, password=VALID_PASSWORD):
        return self._client.post('/auth/login', data={'username': username, 'password': password})

    def logout(self):
        return self._client.get('/auth/logout')


@pytest.fixture
def auth(client):
    return AuthActions(client)
