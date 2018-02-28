import io

import pytest

from pythia.streams.shape_shift_rates import ShapeShiftRates
from pythia.tests.doubles import PairEntryStub


class RecordsStub(io.StringIO):
    def add_record(self, *pairs):
        string = "{2018-02-23 07:55:01.668919: b'[" + ",".join(map(str, pairs)) + "]'}\n"
        self.write(string)
        return self

    def finish(self):
        self.seek(0)


@pytest.fixture
def empty():
    stream = io.StringIO("")
    yield ShapeShiftRates(stream)
    stream.close()


@pytest.fixture
def stream():
    stream = RecordsStub()
    yield stream
    stream.close()


def test_empty(empty):
    with pytest.raises(StopIteration) as e:
        unused = next(iter(empty))


def test_one_pair_entry(stream):
    pair = PairEntryStub("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)
    stream.add_record(pair).finish()
    single = ShapeShiftRates(stream)
    pairs = next(iter(single))
    assert pairs["ETH_SALT"] == pair


def test_multiple_pair_entries(stream):
    pairA = PairEntryStub("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)
    pairB = PairEntryStub("RCN_1ST", 1.52, 0.1, 4.1, 1.1, 1.9)
    stream.add_record(pairA, pairB).finish()
    single = ShapeShiftRates(stream)
    pairs = next(iter(single))
    assert pairs["ETH_SALT"] == pairA
    assert pairs["RCN_1ST"] == pairB


def test_multiple_records(stream):
    pair1 = PairEntryStub("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)
    pair2 = PairEntryStub("ETH_SALT", 1.2, 0.5, 6.7, 0.3, 0.9)
    stream.add_record(pair1).add_record(pair2).finish()
    multi = ShapeShiftRates(stream)
    pairs1 = next(iter(multi))
    pairs2 = next(iter(multi))
    assert pairs1["ETH_SALT"] == pair1
    assert pairs2["ETH_SALT"] == pair2
