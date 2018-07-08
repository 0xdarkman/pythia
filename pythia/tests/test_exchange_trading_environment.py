import io
import pytest

from pythia.core.environment.exchange_trading_environment import ExchangeTradingEnvironment, EnvironmentFinished
from pythia.core.streams.shape_shift_rates import ShapeShiftRates
from pythia.tests.crypto_doubles import RecordsStub, RatesStub, entry


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


def make_env(rates, start_coin="BTC", start_amount=1, window=1, transform=None, reward_calculator=None):
    return ExchangeTradingEnvironment(rates, start_coin, start_amount, window, state_transform=transform,
                                      reward_calculator=reward_calculator)


@pytest.fixture
def two_steps(rates):
    rates.add_record(entry("BTC_ETH", 1), entry("ETH_BTC", 1)) \
        .add_record(entry("BTC_ETH", 2), entry("ETH_BTC", 0.5)).finish()
    yield make_env(rates)
    rates.close()


@pytest.fixture
def three_steps(rates):
    rates.add_record(entry("BTC_ETH", 1), entry("ETH_BTC", 1)) \
        .add_record(entry("BTC_ETH", 2), entry("ETH_BTC", 0.5)) \
        .add_record(entry("BTC_ETH", 3), entry("ETH_BTC", 0.333333)).finish()
    yield make_env(rates)
    rates.close()


@pytest.fixture
def listener():
    return ExchangeListenerSpy()


def test_empty(empty):
    with pytest.raises(EnvironmentFinished) as e_info:
        make_env(empty)
    assert str(e_info.value) == "A rates environment needs at least 2 entries to be initialised."


def test_step_yields_next_state(rates):
    rates.add_record(entry("BTC_ETH", 1.1)) \
        .add_record(entry("BTC_ETH", 1.2)).finish()

    env = make_env(rates)
    s, _, _, _ = env.step(None)
    assert s["token"] == "BTC"
    assert s["balance"] == 1
    assert s["rates"][0]["BTC_ETH"] == entry("BTC_ETH", 1.2)


def test_reset_returns_environment_to_start(rates):
    rates.add_record(entry("BTC_ETH", 1.1)) \
        .add_record(entry("BTC_ETH", 1.2)).finish()
    env = make_env(rates)
    env.step(None)

    env.reset()
    s, _, _, _ = env.step(None)
    assert s["token"] == "BTC"
    assert s["balance"] == 1
    assert s["rates"][0]["BTC_ETH"] == entry("BTC_ETH", 1.2)


def test_reset_returns_first_state(rates):
    rates.add_record(entry("BTC_ETH", 1.1)) \
        .add_record(entry("BTC_ETH", 1.2)).finish()
    env = make_env(rates)
    assert env.reset()["rates"][0]["BTC_ETH"] == entry("BTC_ETH", 1.1)


def test_reset_sets_coin_to_start_coin(three_steps):
    three_steps.step("ETH")
    three_steps.reset()
    three_steps.step("ETH")


def test_reset_sets_amount_to_start_amount(rates):
    rates.add_record(entry("BTC_ETH", 2), entry("ETH_BTC", 0.5)) \
        .add_record(entry("BTC_ETH", 3), entry("ETH_BTC", 0.33)).finish()
    env = make_env(rates, start_amount=2)
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
    assert str(e.value) == "Rates environment finished. No further steps possible."


def test_exchanging(rates):
    rates.add_record(entry("BTC_ETH", "12", "0.002"), entry("ETH_BTC", "0.12")) \
        .add_record(entry("BTC_ETH", "14"), entry("ETH_BTC", "0.14")).finish()
    env = make_env(rates, "BTC", 2)
    env.step("ETH")
    assert env.amount == 23.998
    assert env.token == "ETH"


def test_exchanging_to_itself(rates):
    rates.add_record(entry("BTC_ETH", "12", "0.002")) \
        .add_record(entry("BTC_ETH", "14")).finish()
    env = make_env(rates, "BTC", 2)
    env.step("BTC")
    assert env.amount == 2
    assert env.token == "BTC"


def test_exchange_back_and_forth(rates):
    rates.add_record(entry("BTC_ETH", "12", "0.002"), entry("ETH_BTC", "0.083", "0.002")) \
        .add_record(entry("BTC_ETH", "10", "0.002"), entry("ETH_BTC", "0.1", "0.00006")) \
        .add_record(entry("BTC_ETH", "11"), entry("ETH_BTC", "0.09", "0.00006")).finish()
    env = make_env(rates, "BTC", 2)
    env.step("ETH")
    env.step("BTC")
    assert env.amount == pytest.approx(2.39974)
    assert env.token == "BTC"


def test_initial_balance(rates):
    rates.add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).finish()
    env = make_env(rates, start_coin="BTC", start_amount=2)
    assert env.balance_in("BTC") == 2
    assert env.balance_in("ETH") == 4


def test_current_balance(rates):
    rates.add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).finish()
    env = make_env(rates, start_coin="BTC", start_amount=2)
    env.step(None)
    assert env.balance_in("BTC") == 2
    assert env.balance_in("ETH") == 6


def test_balance_after_exchange(rates):
    rates.add_record(entry("BTC_ETH", "2", miner_fee="0.1")).add_record(entry("ETH_BTC", "1")).finish()
    env = make_env(rates, start_coin="BTC", start_amount=2)
    env.step("ETH")
    assert env.balance_in("BTC") == 3.9
    assert env.balance_in("ETH") == 3.9


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


def test_environment_with_window(rates):
    rates.add_record(entry("BTC_ETH", 1.1)) \
        .add_record(entry("BTC_ETH", 1.2)) \
        .add_record(entry("BTC_ETH", 1.3)).finish()
    s = make_env(rates, window=2).reset()
    assert s["rates"] == [{"BTC_ETH": entry("BTC_ETH", 1.1)}, {"BTC_ETH": entry("BTC_ETH", 1.2)}]


def test_window_moves_when_stepping(rates):
    rates.add_record(entry("BTC_ETH", 1.1)) \
        .add_record(entry("BTC_ETH", 1.2)) \
        .add_record(entry("BTC_ETH", 1.3)).finish()
    e = make_env(rates, window=2)
    e.reset()
    s, _, done, _ = e.step(None)
    assert done
    assert s["rates"] == [{"BTC_ETH": entry("BTC_ETH", 1.2)}, {"BTC_ETH": entry("BTC_ETH", 1.3)}]


def test_transform_rates_if_specified(rates):
    rates.add_record(entry("BTC_ETH", 1.1)) \
        .add_record(entry("BTC_ETH", 1.2)).finish()

    env = make_env(rates, transform=lambda s: float(s["rates"][0]["BTC_ETH"].rate))
    s = env.reset()
    assert s == 1.1
    s, _, _, _ = env.step(None)
    assert s == 1.2


def test_calculates_rate_if_calculator_is_provided(rates):
    rates.add_record(entry("BTC_ETH", 1.1)) \
        .add_record(entry("BTC_ETH", 1.2)).finish()

    env = make_env(rates, reward_calculator=lambda e: 2.0)
    _, r, _, _ = env.step(None)
    assert r == 2.0


def test_reward_calculator_is_called_with_the_current_state(rates):
    rates.add_record(entry("BTC_ETH", 1.1)) \
        .add_record(entry("BTC_ETH", 1.2)).finish()

    class RCalc:
        def __init__(self):
            self.received_state = None

        def __call__(self, state):
            self.received_state = state
            return 0

    calc = RCalc()
    env = make_env(rates, reward_calculator=calc)
    s, _, _, _ = env.step(None)
    assert calc.received_state == s
