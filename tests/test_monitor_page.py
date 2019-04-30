def test_login_required(client):
    assert client.get('/monitor/').headers['Location'] == 'http://localhost/auth/login'


def test_contains_pythia_service(client, auth):
    auth.login()
    response = client.get('/monitor/')
    assert b"Pythia Service" in response.data
