from pytest import fixture

from pythia.core.environment.environment_wrappers import TradingEnvironmentTableWrapper
from .common_fixtures import model_path


@fixture
def env():
    return TradingEnvironmentTableWrapper(1000.0, model_path(), 0)


def make_state(price, step, buying_price):
    return [float(price), float(buying_price), float(step)]


def test_reset_state(env):
    env.step(0)
    s = env.reset()
    assert s == make_state(3, 0, 0)


def test_step_state_hold(env):
    s, _, _, _ = env.step(0)
    assert s == make_state(5, 1, 0)


def test_record_buying_price(env):
    env.step(1)
    env.step(0)
    s, _, _, _, = env.step(0)
    assert s == make_state(1, 3, 3)


def test_sequential_buying(env):
    env.step(1)
    s, _, _, _, = env.step(1)
    assert s == make_state(4, 2, 3)


def test_selling_state(env):
    env.step(1)
    s, _, _, _, = env.step(2)
    assert s == make_state(4, 2, 0)


def test_selling_state_without_shares(env):
    s, _, _, _ = env.step(2)
    assert s == make_state(5, 1, 0)


def test_buying_and_holding_rewards(env):
    _, r, _, _ = env.step(0)
    assert r == 0.0
    _, r, _, _ = env.step(1)
    assert r == 0.0


def test_selling_rewards(env):
    env.step(1)
    _, r, _, _ = env.step(2)
    assert r == 2.0


def test_reward_when_selling_without_shares(env):
    _, r, _, _ = env.step(2)
    assert r == 0.0
    _, r, _, _ = env.step(2)
    assert r == 0.0


def test_trading_episode(env):
    s = env.reset()
    assert s == make_state(3, 0, 0)
    s, r, _, _ = env.step(0)
    assert s == make_state(5, 1, 0)
    assert r == 0.0
    s, r, _, _ = env.step(1)
    assert s == make_state(4, 2, 5)
    assert r == 0.0
    s, r, _, _ = env.step(2)
    assert s == make_state(1, 3, 0)
    assert r == -1.0
    s, r, _, _ = env.step(1)
    assert s == make_state(2, 4, 1)
    assert r == 0.0
    s, r, _, _ = env.step(0)
    assert s == make_state(5, 5, 1)
    assert r == 0.0
    s, r, _, _ = env.step(1)
    assert s == make_state(6, 6, 1)
    assert r == 0.0
    s, r, d, _ = env.step(2)
    assert s == make_state(4, 7, 0)
    assert r == 5.0
    assert d is True


def test_penalize_invalid_actions():
    e = TradingEnvironmentTableWrapper(1000.0, model_path(), 10)
    _, r, _, _ = e.step(1)
    assert r == 0.0
    _, r, _, _ = e.step(1)
    assert r == -10.0
    _, r, _, _ = e.step(2)
    assert r == 1.0
    _, r, _, _ = e.step(2)
    assert r == -10.0
