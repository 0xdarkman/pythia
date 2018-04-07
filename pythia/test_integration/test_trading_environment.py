import inspect
import os

from pytest import approx, fixture

from pythia.core.environment.environment_wrappers import TradingEnvironment, MINIMUM_ACTION
from .common_fixtures import stock_path


@fixture
def environment():
    return TradingEnvironment(1000.0, stock_path())


def test_first_step_holding_action(environment):
    new_state, reward, done, info = environment.step(0.0)

    assert new_state.tolist() == [4.25, 4.31, 3.90, 4.09, 272321.0, 0, 0]
    assert info == 1000.0
    assert done is False


def test_first_step_selling_action(environment):
    new_state, _, _, info = environment.step(-1.0)

    assert info == 1000.0
    assert new_state.tolist() == [4.25, 4.31, 3.90, 4.09, 272321.0, 0, 0]


def test_first_step_buying_action(environment):
    new_state, _, _, info = environment.step(1.0)

    assert info == approx(2.76)
    assert new_state.tolist() == [4.25, 4.31, 3.90, 4.09, 272321.0, 233, 4.28]


def test_keeps_buying_price(environment):
    environment.step(1.0)
    new_state, _, _, _ = environment.step(0.0)

    assert new_state.tolist() == [4.17, 4.21, 4.05, 4.21, 126586, 233, 4.28]


def test_buy_percentage(environment):
    _, _, _, info = environment.step(0.5)
    assert info == 503.52


def test_do_not_buy_at_minimum_action(environment):
    _, _, _, info = environment.step(MINIMUM_ACTION)
    assert info == 1000.0


def test_step_moves_forward_in_time(environment):
    environment.step(0.0)

    new_state, _, _, _, = environment.step(0.0)

    assert new_state.tolist() == [4.17, 4.21, 4.05, 4.21, 126586.0, 0, 0]


def test_selling_with_loss(environment):
    environment.step(1.0)

    _, _, _, info = environment.step(-1.0)

    assert info == 993.01


def test_selling_with_profit(environment):
    environment.step(0.0)
    environment.step(0.0)
    environment.step(1.0)

    _, _, _, info = environment.step(-1.0)

    assert info == 1021.51


def step_right_before_the_end(environment, start_index=0):
    for _ in range(start_index, len(environment.stock_data.data) - 2):
        environment.step(0.0)


def test_done_when_time_series_ends(environment):
    step_right_before_the_end(environment)

    _, _, done, _ = environment.step(0.0)

    assert done is True


def test_reset(environment):
    environment.step(0.0)
    environment.step(1.0)
    environment.step(-1.0)
    environment.step(1.0)

    first_state = environment.reset()

    assert first_state.tolist() == [4.28, 4.38, 4.15, 4.25, 101970, 0, 0]
    assert environment.wealth == environment.portfolio
    assert environment.previous_wealth == environment.portfolio
    assert 0 == len(environment.actions)
    assert 0 == environment.buying_price


def test_selling_at_threshold(environment):
    environment.step(1.0)
    _, _, _, info = environment.step(-MINIMUM_ACTION)
    assert info == approx(2.76)


def test_positive_reward_when_finished(environment):
    environment.step(0.0)
    environment.step(0.0)
    environment.step(1.0)

    _, reward, _, _ = environment.step(-1.0)

    assert reward == approx(21.51)


def test_negative_reward_when_finished(environment):
    environment.step(1.0)

    _, reward, _, _ = environment.step(-1.0)

    assert reward == approx(-6.99)


def test_buying_has_no_reward(environment):
    _, reward, _, _ = environment.step(1.0)
    assert reward == 0.0
