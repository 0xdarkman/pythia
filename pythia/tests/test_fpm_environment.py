import pytest
import numpy as np

from collections import deque

from pythia.tests.fpm_doubles import Prices


class FpmEnvironment:
    def __init__(self, time_series, config):
        self.time_series = time_series
        self.cash = config["trading"]["cash_amount"]
        self.next_prices = None

    def reset(self):
        self.time_series.reset()
        try:
            s = next(self.time_series)
            self.next_prices = next(self.time_series)
            return s
        except StopIteration:
            raise self.TimeSeriesError("The time series provided is empty.")

    def step(self, action):
        r = self._calc_reward_from(action)
        return self._make_next_state(self.next_prices, r)

    def _calc_reward_from(self, action):
        y = np.array(self.next_prices)
        y = self._add_cash_prices(y)
        return np.dot(action, y[:, 0]) * self.cash

    @staticmethod
    def _add_cash_prices(y):
        return np.insert(y, 0, np.ones(y.shape[1]), axis=0)

    def _make_next_state(self, current, reward):
        self.next_prices = next(self.time_series, None)
        return current, reward, self.next_prices is None, None

    class TimeSeriesError(AttributeError):
        pass


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
    return {"trading": {"cash_amount": 100}}


@pytest.fixture
def starting_cash(config):
    return config["trading"]["cash_amount"]


@pytest.fixture
def env(series, config):
    series.set_prices(*(prices(i) for i in range(1, 100)))
    return FpmEnvironment(series, config)


def prices(closing):
    return Prices({"SYM1": {"high": 0.0, "low": 0.0, "close": closing}}).to_array()


def action(portfolio=None):
    return np.array(portfolio if portfolio is not None else [1.0, 0.0])


def prep_env_series(env, series, *closings):
    series.set_prices(*(prices(i) for i in closings))
    env.reset()


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
    _, r, _, _ = env.step(action([1.0, 0.0]))
    assert r == starting_cash


def test_reward_increases_when_price_of_invested_asset_rises(env, series, starting_cash):
    prep_env_series(env, series, 1, 2)
    _, r, _, _ = env.step(action([0.0, 1.0]))
    assert r == starting_cash * 2
