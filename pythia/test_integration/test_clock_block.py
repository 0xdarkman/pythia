import time

from pythia.core.utils.profiling import clock_block


class PrintSpy:
    def __init__(self):
        self.received_string = ""

    def write(self, string):
        self.received_string += string


def test_clocking_time_in_with_block():
    log = PrintSpy()
    with clock_block("Block name", log):
        time.sleep(0.1)
    assert log.received_string.startswith("Block name execution time: 0.")
