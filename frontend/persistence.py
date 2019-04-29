import json
import os

from flask import current_app, g


def get_static():
    if 'static_data' not in g:
        with open(os.path.join(current_app.config['DATA_DIR'], "static.json"), 'r') as f:
            g.static_data = json.load(f)

    return g.static_data
