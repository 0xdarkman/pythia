import pytest


def test_login_catches_invalid_input(client):
    response = client.post('/auth/login', data={'username': '', 'password': 'test'})
    assert b"Invalid login" in response.data


@pytest.mark.skip("WIP - need db first")
def test_successful_login_redirects_to_monitor(client):
    response = client.post('/auth/login', data={'username': 'test', 'password': 'test'})
    assert response.headers['Location'] == "http://localhost/monitor"
