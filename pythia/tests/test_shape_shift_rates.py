import io

import pytest

from pythia.core.streams.shape_shift_rates import ShapeShiftRates
from pythia.tests.doubles import PairEntryStub, RecordsStub


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


def entry(pair, rate, limit, maxLimit, min, minerFee):
    return PairEntryStub(pair, rate, limit, maxLimit, min, minerFee)


def test_empty(empty):
    with pytest.raises(StopIteration) as e:
        unused = next(iter(empty))


def test_one_pair_entry(stream):
    stream.add_record(entry("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)).finish()
    single = ShapeShiftRates(stream)
    assert next(iter(single))["ETH_SALT"] == entry("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)


def test_multiple_pair_entries(stream):
    stream.add_record(entry("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5),
                      entry("RCN_1ST", 1.52, 0.1, 4.1, 1.1, 1.9)).finish()
    single = ShapeShiftRates(stream)
    pairs = next(iter(single))
    assert pairs["ETH_SALT"] == entry("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)
    assert pairs["RCN_1ST"] == entry("RCN_1ST", 1.52, 0.1, 4.1, 1.1, 1.9)


def test_multiple_records(stream):
    stream.add_record(entry("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)) \
        .add_record(entry("ETH_SALT", 1.2, 0.5, 6.7, 0.3, 0.9)).finish()
    multi = ShapeShiftRates(stream)
    assert next(iter(multi))["ETH_SALT"] == entry("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)
    assert next(iter(multi))["ETH_SALT"] == entry("ETH_SALT", 1.2, 0.5, 6.7, 0.3, 0.9)


def test_reset(stream):
    stream.add_record(entry("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)) \
          .add_record(entry("ETH_SALT", 1.2, 0.5, 6.7, 0.3, 0.9)).finish()
    multi = ShapeShiftRates(stream)
    move_to_end(multi)
    multi.reset()
    assert next(iter(multi))["ETH_SALT"] == entry("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)


def move_to_end(multi):
    next(iter(multi))
    next(iter(multi))
