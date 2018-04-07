import io

import pytest

from pythia.core.streams.shape_shift_rates import ShapeShiftRates, rates_filter, interim_lookahead
from pythia.tests.crypto_doubles import PairEntryStub, RecordsStub


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


@pytest.fixture
def out_stream():
    s = io.StringIO()
    yield s
    s.close()


def entry(pair, rate, limit, maxLimit, min, minerFee):
    return PairEntryStub(pair, rate, limit, maxLimit, min, minerFee)


def test_empty(empty):
    with pytest.raises(StopIteration):
        next(iter(empty))


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


def test_empty_filter(stream):
    stream.add_record(entry("ETH_BTC", "1.1", 0.7, 6.2, 0.1, "0.5"),
                      entry("ETH_1ST", "1.5", 0.1, 4.1, 1.1, "1.9")).finish()
    filtered = ShapeShiftRates(io.StringIO(rates_filter(ShapeShiftRates(stream), [])))
    pairs = next(iter(filtered))
    assert pairs["ETH_BTC"] == entry("ETH_BTC", "1.1", 0.7, 6.2, 0.1, "0.5")
    assert pairs["ETH_1ST"] == entry("ETH_1ST", "1.5", 0.1, 4.1, 1.1, "1.9")


def test_filter_exchange(stream):
    stream.add_record(entry("ETH_BTC", "1.1", 0.7, 6.2, 0.1, "0.5"),
                      entry("ETH_1ST", "1.5", 0.1, 4.1, 1.1, "1.9")).finish()
    filtered = ShapeShiftRates(io.StringIO(rates_filter(ShapeShiftRates(stream), ["ETH_1ST"])))
    pairs = next(iter(filtered))
    assert pairs["ETH_1ST"] == entry("ETH_1ST", "1.5", 0.1, 4.1, 1.1, "1.9")
    assert "ETH_BTC" not in pairs


def test_filter_exchange_of_multiple_records(stream):
    stream.add_record(entry("ETH_BTC", "1.1", 0.7, 6.2, 0.1, "0.5"),
                      entry("ETH_1ST", "1.5", 0.1, 4.1, 1.1, "1.9"))\
          .add_record(entry("ETH_BTC", "1.2", 0.7, 6.2, 0.1, "0.4"),
                      entry("BTC_ETH", "1.0", 0.7, 6.2, 0.1, "0.4"),
                      entry("ETH_1ST", "1.4", 0.1, 4.1, 1.1, "1.8"))\
          .add_record(entry("ETH_BTC", "1.3", 0.7, 6.2, 0.1, "0.3"),
                      entry("BTC_ETH", "1.1", 0.7, 6.2, 0.1, "0.5")).finish()
    filtered = ShapeShiftRates(io.StringIO(rates_filter(ShapeShiftRates(stream), ["ETH_1ST"])))
    pairs = next(iter(filtered))
    assert pairs["ETH_1ST"] == entry("ETH_1ST", "1.5", 0.1, 4.1, 1.1, "1.9")
    assert "ETH_BTC" not in pairs
    pairs = next(iter(filtered))
    assert pairs["ETH_1ST"] == entry("ETH_1ST", "1.4", 0.1, 4.1, 1.1, "1.8")
    assert "ETH_BTC" not in pairs
    assert "BTC_ETH" not in pairs
    with pytest.raises(StopIteration):
        next(iter(filtered))


def test_preloaded(stream):
    stream.add_record(entry("ETH_BTC", "1.1", 0.7, 6.2, 0.1, "0.5"),
                      entry("ETH_1ST", "1.5", 0.1, 4.1, 1.1, "1.9"))\
          .add_record(entry("ETH_BTC", "1.2", 0.7, 6.2, 0.1, "0.4"),
                      entry("ETH_1ST", "1.4", 0.1, 4.1, 1.1, "1.8"))\
          .add_record(entry("ETH_BTC", "1.3", 0.7, 6.2, 0.1, "0.3"),
                      entry("ETH_1ST", "1.1", 0.1, 4.1, 1.1, "0.5")).finish()

    rates = ShapeShiftRates(stream, preload=True)
    pairs = next(iter(rates))
    assert pairs["ETH_BTC"] == entry("ETH_BTC", "1.1", 0.7, 6.2, 0.1, "0.5")
    assert pairs["ETH_1ST"] == entry("ETH_1ST", "1.5", 0.1, 4.1, 1.1, "1.9")
    pairs = next(iter(rates))
    assert pairs["ETH_BTC"] == entry("ETH_BTC", "1.2", 0.7, 6.2, 0.1, "0.4")
    assert pairs["ETH_1ST"] == entry("ETH_1ST", "1.4", 0.1, 4.1, 1.1, "1.8")
    pairs = next(iter(rates))
    assert pairs["ETH_BTC"] == entry("ETH_BTC", "1.3", 0.7, 6.2, 0.1, "0.3")
    assert pairs["ETH_1ST"] == entry("ETH_1ST", "1.1", 0.1, 4.1, 1.1, "0.5")
    with pytest.raises(StopIteration):
        next(iter(rates))


@pytest.mark.parametrize("should_preload", [False, True])
def test_interim_look_ahead(stream, should_preload):
    stream.add_record(entry("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5))\
        .add_record(entry("ETH_SALT", 1.2, 0.5, 6.7, 0.3, 0.9))\
        .add_record(entry("ETH_SALT", 1.3, 0.6, 6.3, 0.2, 0.7)).finish()

    rates = ShapeShiftRates(stream, preload=should_preload)
    next(rates)
    with interim_lookahead(rates):
        assert next(rates)["ETH_SALT"] == entry("ETH_SALT", 1.2, 0.5, 6.7, 0.3, 0.9)
        assert next(rates)["ETH_SALT"] == entry("ETH_SALT", 1.3, 0.6, 6.3, 0.2, 0.7)
    assert next(rates)["ETH_SALT"] == entry("ETH_SALT", 1.2, 0.5, 6.7, 0.3, 0.9)
    assert next(rates)["ETH_SALT"] == entry("ETH_SALT", 1.3, 0.6, 6.3, 0.2, 0.7)