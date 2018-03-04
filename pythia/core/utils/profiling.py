import time
from contextlib import contextmanager


@contextmanager
def clock_block(name, out_stream=None):
    start = time.perf_counter()
    yield
    t = time.perf_counter() - start
    print("{} execution time: {}".format(name, t), file=out_stream)