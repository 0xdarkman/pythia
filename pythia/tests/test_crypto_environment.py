import io
import pytest
from decimal import Decimal

from pythia.core.environment.crypto_environment import CryptoEnvironment, EnvironmentFinished
from pythia.core.streams.shape_shift_rates import ShapeShiftRates
from pythia.tests.doubles import RecordsStub, RatesStub, entry


class ExchangeListenerSpy:
    def __init__(self):
        self.received_time = None
        self.received_exchange = None

    def __call__(self, time, exchange):
        self.received_time = time
        self.received_exchange = exchange


@pytest.fixture
def empty():
    stream = io.StringIO("")
    yield ShapeShiftRates(stream)
    stream.close()


@pytest.fixture
def rates():
    s = RecordsStub()
    yield RatesStub(s)
    s.close()


def make_env(rates, start_coin="BTC", start_amount="1"):
    return CryptoEnvironment(rates, start_coin, start_amount)


@pytest.fixture
def two_steps(rates):
    rates.add_record(entry("BTC_ETH", "1")) \
        .add_record(entry("BTC_ETH", "2")).finish()
    yield make_env(rates)
    rates.close()


@pytest.fixture
def three_steps(rates):
    rates.add_record(entry("BTC_ETH", "1")) \
        .add_record(entry("BTC_ETH", "2")) \
        .add_record(entry("BTC_ETH", "3")).finish()
    yield make_env(rates)
    rates.close()


@pytest.fixture
def listener():
    return ExchangeListenerSpy()


def test_empty(empty):
    with pytest.raises(EnvironmentFinished) as e_info:
        make_env(empty)
    assert str(e_info.value) == "A Crypto environment needs at least 2 entries to be initialised."


def test_step_yields_next_state(rates):
    rates.add_record(entry("BTC_ETH", "1.1")) \
        .add_record(entry("BTC_ETH", "1.2")).finish()

    env = make_env(rates)
    s, _, _, _ = env.step(None)
    assert s[0] == "BTC"
    assert s[1]["BTC_ETH"] == entry("BTC_ETH", "1.2")


def test_reset_returns_environment_to_start(rates):
    rates.add_record(entry("BTC_ETH", "1.1")) \
        .add_record(entry("BTC_ETH", "1.2")).finish()
    env = make_env(rates)
    env.step(None)

    env.reset()
    s, _, _, _ = env.step(None)
    assert s[0] == "BTC"
    assert s[1]["BTC_ETH"] == entry("BTC_ETH", "1.2")


def test_reset_returns_first_state(rates):
    rates.add_record(entry("BTC_ETH", "1.1")) \
        .add_record(entry("BTC_ETH", "1.2")).finish()
    env = make_env(rates)
    assert env.reset()[1]["BTC_ETH"] == entry("BTC_ETH", "1.1")


def test_reset_sets_coin_to_start_coin(three_steps):
    three_steps.step("ETH")
    three_steps.reset()
    three_steps.step("ETH")


def test_reset_sets_amount_to_start_amount(rates):
    rates.add_record(entry("BTC_ETH", "2")) \
        .add_record(entry("BTC_ETH", "3")).finish()
    env = make_env(rates, start_amount="2")
    env.step("ETH")
    env.reset()
    assert env.amount == 2



def test_last_step_yields_done_true(two_steps):
    _, _, done, _ = two_steps.step(None)
    assert done


def test_in_between_step_yields_done_false(three_steps):
    _, _, done, _ = three_steps.step(None)
    assert done is False


def test_stepping_past_done_state(two_steps):
    two_steps.step(None)
    with pytest.raises(EnvironmentFinished) as e:
        two_steps.step(None)
    assert str(e.value) == "CryptoEnvironment finished. No further steps possible."


def test_exchanging(rates):
    rates.add_record(entry("BTC_ETH", "12", "0.002")) \
        .add_record(entry("BTC_ETH", "14")).finish()
    env = make_env(rates, "BTC", "2")
    env.step("ETH")
    assert env.amount == Decimal("23.998")
    assert env.coin == "ETH"


def test_exchanging_to_itself(rates):
    rates.add_record(entry("BTC_ETH", "12", "0.002")) \
        .add_record(entry("BTC_ETH", "14")).finish()
    env = make_env(rates, "BTC", "2")
    env.step("BTC")
    assert env.amount == Decimal("2")
    assert env.coin == "BTC"


def test_exchange_back_and_forth(rates):
    rates.add_record(entry("BTC_ETH", "12", "0.002")) \
        .add_record(entry("ETH_BTC", "0.1", "0.00006")) \
        .add_record(entry("BTC_ETH", "11")).finish()
    env = make_env(rates, "BTC", "2")
    env.step("ETH")
    env.step("BTC")
    assert env.amount == Decimal("2.39974")
    assert env.coin == "BTC"


def test_initial_balance(rates):
    rates.add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).finish()
    env = make_env(rates, start_coin="BTC", start_amount=Decimal(2))
    assert env.balance_in("BTC") == Decimal("2")
    assert env.balance_in("ETH") == Decimal("4")


def test_current_balance(rates):
    rates.add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).finish()
    env = make_env(rates, start_coin="BTC", start_amount=Decimal(2))
    env.step(None)
    assert env.balance_in("BTC") == Decimal("2")
    assert env.balance_in("ETH") == Decimal("6")


def test_balance_after_exchange(rates):
    rates.add_record(entry("BTC_ETH", "2", miner_fee="0.1")).add_record(entry("ETH_BTC", "1")).finish()
    env = make_env(rates, start_coin="BTC", start_amount=Decimal(2))
    env.step("ETH")
    assert env.balance_in("BTC") == Decimal("3.9")
    assert env.balance_in("ETH") == Decimal("3.9")


def test_notify_listeners_when_exchanging(three_steps, listener):
    three_steps.register_listener(listener)
    three_steps.step(None)

    assert listener.received_time is None
    assert listener.received_exchange is None

    three_steps.step("ETH")

    assert listener.received_time == 1
    assert listener.received_exchange == "ETH"


def test_time_resets(three_steps, listener):
    three_steps.register_listener(listener)
    three_steps.step("ETH")

    three_steps.reset()
    three_steps.step("ETH")

    assert listener.received_time == 0
