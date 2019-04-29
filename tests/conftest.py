import json
import os
import platform
import shutil
import tempfile

import pytest
from werkzeug.security import generate_password_hash

from frontend import create_app


def make_tmp():
    if platform.system() == 'Linux':
        d = os.path.join("/dev/shm", tempfile.mktemp())
        os.makedirs(d)
        return d
    return tempfile.mkdtemp()


@pytest.fixture
def app():
    tmp = make_tmp()
    with open(os.path.join(tmp, 'static.json'), 'w') as f:
        json.dump({'user': 'test', 'password': generate_password_hash('test')}, f)
    yield create_app({'TESTING': True, 'DATA_DIR': tmp})
    shutil.rmtree(tmp)


@pytest.fixture
def client(app):
    return app.test_client()
