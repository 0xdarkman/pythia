import pytest

from pythia.core.environment.crypto_ai_environment import CryptoAiEnvironment, WindowError
from pythia.core.environment.crypto_rewards import TotalBalanceReward, RatesChangeReward
from pythia.tests.doubles import RecordsStub, RatesStub, entry


class AiRatesStub(RatesStub):
    def add_pairs(self, *pairs):
        for line in zip(*pairs):
            self.add_record(*(entry(str(i), str(rate)) for i, rate in enumerate(line)))
        self.finish()
        return self


class RewardCalculatorStub:
    def __init__(self, reward):
        self.reward = reward

    def __call__(self, env):
        return self.reward


class RewardCalculatorSpy:
    def __init__(self):
        self.received_arg = None

    def __call__(self, env):
        self.received_arg = env


@pytest.fixture
def rates():
    s = RecordsStub()
    yield AiRatesStub(s)
    s.close()


@pytest.fixture
def calc_spy():
    return RewardCalculatorSpy()


def make_env(rates, start_coin="0", start_amount="1", window=1, exchange_filter=None, index_to_coin=None,
             reward_calc=None):
    index_to_coin = {0: '0', 1: '1'} if index_to_coin is None else index_to_coin
    reward_calc = RewardCalculatorStub(0) if reward_calc is None else reward_calc
    return CryptoAiEnvironment(rates, start_coin, start_amount, window, index_to_coin, reward_calc, exchange_filter)


def test_data_too_small_for_window(rates):
    rates.add_record(entry("BTC_ETH", "1")).add_record(entry("BTC_ETH", "2")).finish()
    with pytest.raises(WindowError) as e:
        make_env(rates, window=3)
    assert str(e.value) == "There is not enough data to fill the window of size 3"


def test_rates_are_normalized(rates):
    rates.add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).add_record(entry("BTC_ETH", "1")).finish()
    assert make_env(rates, window=3).reset() == [0, 0.0, 0.5, 1.0, 0.0]


@pytest.mark.parametrize("size, expected_state", [
    (1, [0, 0.0, 0.5]),
    (2, [0, 0.0, 0.5, 1.0]),
    (3, [0, 0.0, 0.5, 1.0, 0.0])
])
def test_window_sizes(rates, size, expected_state):
    rates.add_record(entry("BTC_ETH", "2")).add_record(entry("BTC_ETH", "3")).add_record(entry("BTC_ETH", "1")).finish()
    assert make_env(rates, window=size).reset() == expected_state


def test_multiple_pairs(rates):
    rates.add_pairs((2, 1, 3), (10, 12, 20))
    assert make_env(rates, window=3).reset() == [0, 0.0, 0.5, 0.0, 0.0, 0.2, 1.0, 1.0]


@pytest.mark.parametrize("allowed_rate, expected_state", [
    ('0', [0, 0.0, 0.5, 0.0, 1.0]),
    ('1', [0, 0.0, 0.0, 0.2, 1.0]),
])
def test_filter_pairs(rates, allowed_rate, expected_state):
    rates.add_pairs((2, 1, 3), (10, 12, 20))
    assert make_env(rates, window=3, exchange_filter=[allowed_rate]).reset() == expected_state


def test_step_moves_window(rates):
    rates.add_pairs((2, 1, 3), (10, 12, 20))
    s, _, _, _ = make_env(rates, window=2).step(None)
    assert s[-4:] == [0.0, 0.2, 1.0, 1.0]


def test_last_state_returns_done(rates):
    rates.add_pairs((2, 1, 3), (10, 12, 20))
    _, _, done, _ = make_env(rates, window=2).step(None)
    assert done


def test_exchange_coin(rates):
    rates.add_record(entry("BTC_ETH", "2"), entry("ETH_BTC", "10")) \
        .add_record(entry("BTC_ETH", "1"), entry("ETH_BTC", "12")) \
        .add_record(entry("BTC_ETH", "3"), entry("ETH_BTC", "20")).finish()
    s, _, _, _ = make_env(rates, start_coin="BTC", index_to_coin={0: "BTC", 1: "ETH"}).step(1)
    assert s[0] == 1


def test_normalized_balance(rates):
    rates.add_record(entry("BTC_ETH", "2", "0"), entry("ETH_BTC", "0.5", "0")) \
        .add_record(entry("BTC_ETH", "1", "0"), entry("ETH_BTC", "1", "0")).finish()
    s, _, _, _ = make_env(rates, start_coin="BTC", index_to_coin={0: "BTC", 1: "ETH"}).step(1)
    assert s[1] == 1.0


def test_returns_calculated_reward(rates):
    rates.add_pairs((1, 2, 3))
    _, r, _, _ = make_env(rates, reward_calc=RewardCalculatorStub(10.0)).step(None)
    assert r == 10.0


def test_reward_calc_receives_environment(rates, calc_spy):
    env = make_env(rates.add_pairs((1, 2, 3)), reward_calc=calc_spy)
    env.step(None)
    assert calc_spy.received_arg is env


def test_environment_has_last_state(rates):
    env = make_env(rates.add_pairs((1, 2, 3)))
    env.step(None)
    assert env.state == [0, 0.0, 0.5]


@pytest.mark.parametrize("coin_rates, expected_reward", [
    ([(entry("BTC_ETH", "2", "0"), entry("ETH_BTC", "0.5", "0")),
      (entry("BTC_ETH", "1", "0"), entry("ETH_BTC", "1", "0"))], 1.0),
    ([(entry("BTC_ETH", "2", "0"), entry("ETH_BTC", "0.5", "0")),
      (entry("BTC_ETH", "4", "0"), entry("ETH_BTC", "0.25", "0"))], -0.5),
])
def test_total_balance_reward(rates, coin_rates, expected_reward):
    rates.add_record(*coin_rates[0]).add_record(*coin_rates[1]).finish()
    _, r, _, _ = make_env(rates, start_coin="BTC", index_to_coin={0: "BTC", 1: "ETH"},
                          reward_calc=TotalBalanceReward()).step(1)
    assert r == expected_reward


def test_change_in_rates_reward(rates):
    env = make_env(rates.add_pairs((1, 1.5, 1.25, 2.0)), reward_calc=RatesChangeReward())
    assert env.step(None)[1] == 0.5
    assert env.step(None)[1] == -0.25


def test_change_in_rates_reward_takes_rates_change_into_account(rates):
    rates.add_record(entry("BTC_ETH", "2"), entry("ETH_BTC", "10")) \
        .add_record(entry("BTC_ETH", "1"), entry("ETH_BTC", "12")) \
        .add_record(entry("BTC_ETH", "3"), entry("ETH_BTC", "20")).finish()
    env = make_env(rates, start_coin="BTC", index_to_coin={0: "BTC", 1: "ETH"},
                   reward_calc=RatesChangeReward())
    env.step(1)
    assert env.step(None)[1] == 0.8
