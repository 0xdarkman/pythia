import inspect
import os


def model_path():
    dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return os.path.join(dir, "test_data/test_model_data.csv")


def stock_path():
    dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return os.path.join(dir, "test_data/test_stock_data.csv")