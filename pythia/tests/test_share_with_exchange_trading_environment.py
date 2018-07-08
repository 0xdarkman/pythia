import pytest

from io import StringIO

from pythia.core.environment.rates_ai_environment import ExchangeTradingAiEnvironment
from pythia.core.streams.share_rates import ShareRates
from pythia.tests.ai_environment_doubles import RewardCalculatorStub
from pythia.tests.shares_doubles import SymbolStub, entry


class ShareRatesBuilder:
    def __init__(self, symbol_a, symbol_b=None):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.entries_a = []
        self.entries_b = []

    def add_records_a(self, *entries):
        self.entries_a.extend(entries)
        return self

    def add_records_b(self, *entries):
        self.entries_b.extend(entries)
        return self

    def finish(self):
        with self.symbol_a as s:
            for e in self.entries_a:
                self.symbol_a.add_record(e)

        if self.symbol_b is not None:
            with self.symbol_b as s:
                for e in self.entries_b:
                    self.symbol_b.add_record(e)

        return ShareRates(self.symbol_a, self.symbol_b)


@pytest.fixture
def symbol_a():
    with StringIO() as s:
        yield SymbolStub("SYMA", s)


@pytest.fixture
def symbol_b():
    with StringIO() as s:
        yield SymbolStub("SYMB", s)


def rate_entry(rate):
    return entry(rate, rate, rate, rate, 2000)


def make_env(rates, start_token='CURRENCY', window=1, index_to_coin=None):
    index_to_coin = {0: 'CURRENCY', 1: 'SYMA'} if index_to_coin is None else index_to_coin
    reward_calc = RewardCalculatorStub(0)
    return ExchangeTradingAiEnvironment(rates, start_token, 100, window, index_to_coin, reward_calc)


def test_create_rates_environment_with_share_rates(symbol_a):
    rates = ShareRatesBuilder(symbol_a) \
        .add_records_a(rate_entry(2.0), rate_entry(2.0)) \
        .finish()
    make_env(rates)


def test_rates_are_normalized(symbol_a):
    rates = ShareRatesBuilder(symbol_a) \
        .add_records_a(rate_entry(2.0), rate_entry(3.0), rate_entry(1.0)) \
        .finish()
    assert make_env(rates, window=3).reset() == [0, 0.0, 0.5, 0.25, 1.0, 0.0, 0.0, 1.0]


def test_rates_environment_handles_multiple_symbold(symbol_a, symbol_b):
    rates = ShareRatesBuilder(symbol_a, symbol_b) \
        .add_records_a(rate_entry(4.0), rate_entry(9.0), rate_entry(3.0)) \
        .add_records_b(rate_entry(2.0), rate_entry(3.0), rate_entry(3.0)) \
        .finish()
    env = make_env(rates, "SYMA", window=3, index_to_coin={0: 'SYMA', 1: 'SYMB'})
    assert env.reset() == [0, 0.0, 0.5, 0.25, 1.0, 0.0, 0.0, 1.0]


def test_exchange_coin(symbol_a):
    rates = ShareRatesBuilder(symbol_a) \
        .add_records_a(rate_entry(2.0), rate_entry(3.0), rate_entry(1.0)) \
        .finish()
    s, _, _, _ = make_env(rates, start_token="CURRENCY", index_to_coin={0: "CURRENCY", 1: "SYMA"}).step(1)
    assert s[0] == 1


def test_normalized_balance(symbol_a):
    rates = ShareRatesBuilder(symbol_a) \
        .add_records_a(rate_entry(0.5), rate_entry(1)) \
        .finish()
    s, _, _, _ = make_env(rates, start_token="CURRENCY", index_to_coin={0: "CURRENCY", 1: "SYMA"}).step(1)
    assert s[1] == 1.0
