import json
import os

from pythia.core.streams.poloniex_history import PoloniexHistory


def run_fpm(config, data_path, update_history):
    if update_history:
        h = PoloniexHistory(config, data_path)
        h.update()


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "fpm_default.json"), "r") as f:
        cfg = json.load(f)
    run_fpm(cfg, R"D:\Research\AI\MyProjects\Pythia\data\recordings\poloniex", False)
