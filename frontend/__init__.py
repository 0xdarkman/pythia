import os

from flask import Flask


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config['DATA_DIR'] = app.instance_path

    if test_config is None:
        app.config.from_json('config.json', silent=True)
    else:
        app.config['SECRET_KEY'] = 'dev'
        app.config.from_mapping(test_config)

    os.makedirs(app.instance_path, exist_ok=True)

    from . import auth
    app.register_blueprint(auth.bp)

    return app
