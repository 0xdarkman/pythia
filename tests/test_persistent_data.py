from werkzeug.security import check_password_hash

from frontend.persistence import get_static


def test_get_static_data(app, username, password):
    with app.app_context():
        s = get_static()
        assert check_password_hash(s.password, password)
        assert s.user == username


def test_static_data_is_cashed(app):
    with app.app_context():
        assert get_static() is get_static()


def test_loaded_anew_in_each_context(app):
    with app.app_context():
        s = get_static()

    with app.app_context():
        assert s is not get_static()
