from collections import deque

import numpy as np
import pytest

from pythia.core.environment.fpm_environment import FpmEnvironment
from pythia.tests.fpm_doubles import Prices


class PricesTimeSeriesStub:
    def __init__(self):
        self.prices = deque()
        self.iter = None
        self.reset()

    def set_prices(self, *prices):
        self.prices = prices
        self.reset()

    def reset(self):
        self.iter = iter(deque(self.prices))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)


@pytest.fixture
def series():
    return PricesTimeSeriesStub()


@pytest.fixture
def config():
    return {"trading": {"cash_amount": 100, "commission": 0, "coins": ["SYM1"]}}


@pytest.fixture
def starting_cash(config):
    return config["trading"]["cash_amount"]


@pytest.fixture
def env(series, config):
    series.set_prices(*(prices(i) for i in range(1, 100)))
    return FpmEnvironment(series, config)


def prices(*closings):
    entries = dict()
    for i, c in enumerate(closings):
        entries[f"SYM{i}"] = {"high": 0.0, "low": 0.0, "close": c}
    return Prices(entries).to_array()


def action(portfolio=None):
    return np.array(portfolio if portfolio is not None else [1.0, 0.0])


def get_reward(s):
    _, r, _, _ = s
    return r


def prev_env_with_prices(env, series, *p):
    series.set_prices(*p)
    env.reset()


def prep_env_series(env, series, *closings):
    prev_env_with_prices(env, series, *(prices(i) for i in closings))


def test_that_environment_raises_an_error_when_time_series_is_empty(env, series):
    series.set_prices()
    with pytest.raises(FpmEnvironment.TimeSeriesError):
        env.reset()


def test_that_environment_raises_an_error_when_time_series_contains_only_one_element(env, series):
    series.set_prices(prices(1))
    with pytest.raises(FpmEnvironment.TimeSeriesError):
        env.reset()


def test_reset_returns_first_prices_vector(env, series):
    series.set_prices(prices(1), prices(2))
    assert env.reset() == prices(1)


def test_step_returns_next_price_vector(env, series):
    prep_env_series(env, series, 1, 2)
    s, _, _, _ = env.step(action())
    assert s == prices(2)


def test_step_returns_done_if_it_was_last_price_in_time_series(env, series):
    prep_env_series(env, series, 1, 2)
    _, _, done, _ = env.step(action())
    assert done


def test_step_returns_not_done_if_there_are_prices_remaining_in_time_series(env, series):
    prep_env_series(env, series, 1, 2, 3)
    _, _, done, _ = env.step(action())
    assert not done


def test_step_iterates_time_series(env, series):
    prep_env_series(env, series, 1, 2, 3)
    assert env.step(action())[0] == prices(2)
    assert env.step(action())[0] == prices(3)


def test_reset_resets_the_time_series_again(env, series):
    prep_env_series(env, series, 1, 2)
    assert env.step(action())[0] == prices(2)
    assert env.reset() == prices(1)
    assert env.step(action())[0] == prices(2)


def test_trivial_reward_when_action_keeps_cash(env, starting_cash):
    env.reset()
    assert get_reward(env.step(action([1.0, 0.0]))) == starting_cash


def test_no_actions_do_not_change_reward(env, series, starting_cash):
    prep_env_series(env, series, 1, 1, 1)
    assert get_reward(env.step(action([1.0, 0.0]))) == starting_cash
    assert get_reward(env.step(action([1.0, 0.0]))) == starting_cash


def test_reward_increases_when_price_of_invested_asset_rises(env, series, starting_cash):
    prep_env_series(env, series, 1, 2)
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 2


def test_reset_resets_the_portfolio(env, series, starting_cash):
    prep_env_series(env, series, 1, 2)
    env.step(action([0.0, 1.0]))
    env.reset()
    assert (env.assets == [starting_cash, 0]).all()


def test_trading_signals_propagate_correctly_after_reset(env, series, starting_cash):
    prep_env_series(env, series, 1, 2)
    env.step(action([0.0, 1.0]))
    env.reset()
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 2


def test_reward_decreases_when_price_of_invested_asset_sinks(env, series, starting_cash):
    prep_env_series(env, series, 1, 0.5)
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 0.5


def test_reward_is_high_when_buying_low_and_selling_high(env, series, starting_cash):
    prep_env_series(env, series, 0.5, 2, 0.1)
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 4
    assert get_reward(env.step(action([1.0, 0.0]))) == starting_cash * 4


def test_reward_is_low_when_buying_high_and_selling_low(env, series, starting_cash):
    prep_env_series(env, series, 2, 0.5, 4)
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 0.25
    assert get_reward(env.step(action([1.0, 0.0]))) == starting_cash * 0.25


def test_reward_increases_when_holding_assets_and_their_prices_rises(env, series, starting_cash):
    prep_env_series(env, series, 1, 0.5, 2)
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 0.5
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 2


def test_rewards_of_investment_policy(env, series, starting_cash):
    prep_env_series(env, series, 1, 0.5, 2, 1, 0.5, 0.5, 2, 1)
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 0.5
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 2
    assert get_reward(env.step(action([1.0, 0.0]))) == starting_cash * 2
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 1
    assert get_reward(env.step(action([1.0, 0.0]))) == starting_cash * 1
    assert get_reward(env.step(action([1.0, 0.0]))) == starting_cash * 1
    assert get_reward(env.step(action([0.0, 1.0]))) == starting_cash * 0.5


def test_reward_deducts_commission_fee(env, series, starting_cash):
    env.commission = 0.1
    prep_env_series(env, series, 0.5, 2, 0.1)
    assert get_reward(env.step(action([0.0, 1.0]))) == (1 - env.commission) * starting_cash * 4
    assert get_reward(env.step(action([1.0, 0.0]))) == (1 - env.commission) * (1 - env.commission) * starting_cash * 4


def test_reward_is_correctly_calculated_with_multiple_assets(config, series, starting_cash):
    config["trading"]["coins"] = ["SYM1", "SYM2"]
    env = FpmEnvironment(series, config)
    prev_env_with_prices(env, series, prices(0.5, 2.0), prices(2.0, 0.5), prices(0.5, 0.5), prices(1, 2),
                         prices(0.5, 1), prices(0.6, 1.2), prices(1.2, 0.8))
    assert get_reward(env.step(action([0.0, 1.0, 0.0]))) == starting_cash * 4
    assert get_reward(env.step(action([0.0, 0.5, 0.5]))) == starting_cash * 2.5
    assert get_reward(env.step(action([0.5, 0.0, 0.5]))) == starting_cash * 8.5
    assert get_reward(env.step(action([0.5, 0.25, 0.25]))) == starting_cash * 4.5
    assert get_reward(env.step(action([0.6, 0.1, 0.3]))) == starting_cash * 5.14
    assert get_reward(env.step(action([0.3, 0.5, 0.1]))) == starting_cash * 8.35


def test_reward_handles_commission_correctly_with_multiple_assets(config, series, starting_cash):
    config["trading"]["coins"] = ["SYM1", "SYM2"]
    config["trading"]["commission"] = 0.1
    env = FpmEnvironment(series, config)
    prev_env_with_prices(env, series, prices(0.5, 0.5), prices(2.0, 2.0), prices(0.1, 0.1))
    assert get_reward(env.step(action([0.0, 1.0, 0.0]))) == starting_cash * 3.6
