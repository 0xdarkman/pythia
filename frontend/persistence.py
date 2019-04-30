import json
import os
from collections import namedtuple

from flask import current_app, g


def get_static():
    if 'static_data' not in g:
        with open(os.path.join(current_app.config['DATA_DIR'], "static.json"), 'r') as f:
            d = json.load(f)
            g.static_data = namedtuple("StaticData", d.keys())(**d)

    return g.static_data
