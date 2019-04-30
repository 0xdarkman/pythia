import pytest
from flask import g, session


def test_getting_login_request_is_valid(client):
    assert client.get('/auth/login').status_code == 200


@pytest.mark.parametrize('name,pw', (
        ("INVALID", None),
        (None, "INVALID")
))
def test_retry_login_when_invalid(client, auth, name, pw, username, password):
    response = auth.login(name or username, pw or password)
    assert response.data in client.get('/auth/login').data


def test_successful_login_redirects_to_monitor(auth):
    response = auth.login()
    assert response.headers['Location'] == "http://localhost/monitor"


def test_successful_login_generates_valid_session(client, auth):
    auth.login()
    with client:
        client.get('/monitor')
        assert g.valid_session


def test_logout(client, auth):
    auth.login()
    with client:
        auth.logout()
        assert 'date' not in session
        client.get('/monitor')
        assert not g.valid_session
