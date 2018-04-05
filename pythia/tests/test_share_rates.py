import pytest
from io import StringIO

from pythia.core.streams.share_rates import RatesPair, Symbol, ShareRates, interim_lookahead


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
        self.stream = stream
        super().__init__(name, self.stream)

    def __enter__(self):
        self.stream.write(self.HEADER)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.seek(0)

    def add_record(self, entry):
        self.stream.write("2000-01-01 00:00:00,{},{},{},{},{}\n".format(
            entry.open, entry.high, entry.low, entry.close, entry.volume
        ))


@pytest.fixture
def symbol_a():
    with StringIO() as s:
        yield SymbolStub("SYMA", s)


@pytest.fixture
def symbol_b():
    with StringIO() as s:
        yield SymbolStub("SYMB", s)


def entry(*shares_data):
    return RatesPairStub(*shares_data)


def make_symbol(name, stream):
    return Symbol(name, stream)


def make_rates(symbol_a, symbol_b=None):
    return ShareRates(symbol_a, symbol_b)


def test_symbol_empty():
    symbol = make_symbol("SomeName", StringIO())
    with pytest.raises(StopIteration):
        next(symbol)


def test_ignore_header():
    symbol = make_symbol("SomeName", StringIO("timestamp,open,high,low,close,volume"))
    with pytest.raises(StopIteration):
        next(symbol)


def test_return_symbol_market_data():
    symbol = make_symbol("SomeName", StringIO("timestamp,open,high,low,close,volume\n"
                                              "2018-03-20 09:30:00,49.0900,49.1100,49.0700,49.1100,79426"))
    open, high, low, close, volume = next(symbol)
    assert open == 49.09 and high == 49.11 and low == 49.07 and close == 49.11 and volume == 79426


def test_iterate_over_market_data():
    symbol = make_symbol("SomeName", StringIO("timestamp,open,high,low,close,volume\n"
                                              "2018-03-20 09:30:00,49.0900,49.1100,49.0700,49.1100,79426\n"
                                              "2018-03-20 09:31:00,49.1100,49.1400,49.1000,49.1200,173824"))
    next(symbol)
    open, high, low, close, volume = next(symbol)
    assert open == 49.11 and high == 49.14 and low == 49.1 and close == 49.12 and volume == 173824


def test_empty(symbol_a):
    rates = make_rates(symbol_a)
    with pytest.raises(StopIteration):
        next(rates)


def test_one_symbol_one_entry(symbol_a):
    with symbol_a as s:
        s.add_record(entry(1.1, 1.4, 1.0, 1.2, 2100))
    rates = make_rates(symbol_a)
    e = next(rates)
    assert e["SYMA_CURRENCY"] == entry(1.1, 1.4, 1.0, 1.2, 2100)
    assert e["CURRENCY_SYMA"] == entry(1 / 1.1, 1 / 1.4, 1 / 1.0, 1 / 1.2, 2100)


def test_one_symbol_multiple_entries(symbol_a):
    with symbol_a as s:
        s.add_record(entry(1.1, 1.4, 1.0, 1.2, 2100))
        s.add_record(entry(1.2, 1.5, 1.1, 1.3, 2200))
    rates = make_rates(symbol_a)
    next(rates)
    e = next(rates)
    assert e["SYMA_CURRENCY"] == entry(1.2, 1.5, 1.1, 1.3, 2200)
    assert e["CURRENCY_SYMA"] == entry(1 / 1.2, 1 / 1.5, 1 / 1.1, 1 / 1.3, 2200)


def test_two_symbols_one_entry(symbol_a, symbol_b):
    with symbol_a as sa:
        sa.add_record(entry(1.1, 1.4, 1.0, 1.2, 2100))
    with symbol_b as sb:
        sb.add_record(entry(2.1, 2.4, 2.0, 2.2, 4100))
    rates = make_rates(symbol_a, symbol_b)
    e = next(rates)
    assert e["SYMA_SYMB"] == entry(1.1 / 2.1, 1.4 / 2.4, 1.0 / 2.0, 1.2 / 2.2, 2100)
    assert e["SYMB_SYMA"] == entry(2.1 / 1.1, 2.4 / 1.4, 2.0 / 1.0, 2.2 / 1.2, 4100)


def test_two_symbols_multiple_entry(symbol_a, symbol_b):
    with symbol_a as sa:
        sa.add_record(entry(1.1, 1.4, 1.0, 1.2, 2100))
        sa.add_record(entry(1.2, 1.5, 1.1, 1.3, 2200))
        sa.add_record(entry(1.3, 1.6, 1.2, 1.4, 2300))
    with symbol_b as sb:
        sb.add_record(entry(2.1, 2.4, 2.0, 2.2, 4100))
        sb.add_record(entry(2.2, 2.5, 2.1, 2.3, 4200))
    rates = make_rates(symbol_a, symbol_b)
    next(rates)
    e = next(rates)
    assert e["SYMA_SYMB"] == entry(1.2 / 2.2, 1.5 / 2.5, 1.1 / 2.1, 1.3 / 2.3, 2200)
    assert e["SYMB_SYMA"] == entry(2.2 / 1.2, 2.5 / 1.5, 2.1 / 1.1, 2.3 / 1.3, 4200)
    with pytest.raises(StopIteration):
        next(rates)


def test_reset(symbol_a, symbol_b):
    with symbol_a as sa:
        sa.add_record(entry(1.1, 1.4, 1.0, 1.2, 2100))
        sa.add_record(entry(1.2, 1.5, 1.1, 1.3, 2200))
        sa.add_record(entry(1.3, 1.6, 1.2, 1.4, 2300))
    with symbol_b as sb:
        sb.add_record(entry(2.1, 2.4, 2.0, 2.2, 4100))
        sb.add_record(entry(2.2, 2.5, 2.1, 2.3, 4200))
    rates = make_rates(symbol_a, symbol_b)
    next(rates)
    rates.reset()
    e = next(rates)
    assert e["SYMA_SYMB"] == entry(1.1 / 2.1, 1.4 / 2.4, 1.0 / 2.0, 1.2 / 2.2, 2100)
    assert e["SYMB_SYMA"] == entry(2.1 / 1.1, 2.4 / 1.4, 2.0 / 1.0, 2.2 / 1.2, 4100)


def test_interim_look_ahead(symbol_a, symbol_b):
    with symbol_a as sa:
        sa.add_record(entry(1.1, 1.4, 1.0, 1.2, 2100))
        sa.add_record(entry(1.2, 1.5, 1.1, 1.3, 2200))
        sa.add_record(entry(1.3, 1.6, 1.2, 1.4, 2300))
    with symbol_b as sb:
        sb.add_record(entry(2.1, 2.4, 2.0, 2.2, 4100))
        sb.add_record(entry(2.2, 2.5, 2.1, 2.3, 4200))
        sb.add_record(entry(2.3, 2.6, 2.2, 2.4, 4300))
    rates = make_rates(symbol_a, symbol_b)
    next(rates)
    with interim_lookahead(rates):
        assert next(rates)["SYMA_SYMB"] == entry(1.2 / 2.2, 1.5 / 2.5, 1.1 / 2.1, 1.3 / 2.3, 2200)
        assert next(rates)["SYMA_SYMB"] == entry(1.3 / 2.3, 1.6 / 2.6, 1.2 / 2.2, 1.4 / 2.4, 2300)
    assert next(rates)["SYMA_SYMB"] == entry(1.2 / 2.2, 1.5 / 2.5, 1.1 / 2.1, 1.3 / 2.3, 2200)
    assert next(rates)["SYMA_SYMB"] == entry(1.3 / 2.3, 1.6 / 2.6, 1.2 / 2.2, 1.4 / 2.4, 2300)


def test_look_ahead_skips_header_again(symbol_a, symbol_b):
    with symbol_a as sa:
        sa.add_record(entry(1.1, 1.4, 1.0, 1.2, 2100))
    with symbol_b as sb:
        sb.add_record(entry(2.1, 2.4, 2.0, 2.2, 4100))
    rates = make_rates(symbol_a, symbol_b)
    with interim_lookahead(rates):
        assert next(rates)["SYMA_SYMB"] == entry(1.1 / 2.1, 1.4 / 2.4, 1.0 / 2.0, 1.2 / 2.2, 2100)
    assert next(rates)["SYMA_SYMB"] == entry(1.1 / 2.1, 1.4 / 2.4, 1.0 / 2.0, 1.2 / 2.2, 2100)


def test_look_ahead_only_one_symbol(symbol_a):
    with symbol_a as s:
        s.add_record(entry(1.1, 1.4, 1.0, 1.2, 2100))
    rates = make_rates(symbol_a)
    with interim_lookahead(rates):
        assert next(rates)["SYMA_CURRENCY"] == entry(1.1, 1.4, 1.0, 1.2, 2100)
    assert next(rates)["SYMA_CURRENCY"] == entry(1.1, 1.4, 1.0, 1.2, 2100)
