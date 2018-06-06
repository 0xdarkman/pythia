import pytest

from pythia.core.streams.share_rates import RatesPair, Symbol


class RatesPairStub(RatesPair):
    def __eq__(self, other):
        return self.open == pytest.approx(other.open) and \
               self.high == pytest.approx(other.high) and \
               self.low == pytest.approx(other.low) and \
               self.close == pytest.approx(other.close) and \
               self.volume == other.volume


class SymbolStub(Symbol):
    HEADER = "timestamp,open,high,low,close,volume\n"

    def __init__(self, name, stream):
        self.name = name
        self.stream = stream
        super().__init__(self.name, self.stream)

    def __enter__(self):
        self.stream.write(self.HEADER)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.seek(0)
        super().__init__(self.name, self.stream)

    def add_record(self, entry):
        self.stream.write("2000-01-01 00:00:00,{},{},{},{},{}\n".format(
            entry.open, entry.high, entry.low, entry.close, entry.volume
        ))


def entry(*shares_data):
    return RatesPairStub(*shares_data)