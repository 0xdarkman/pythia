import numpy as np
import pytest

from pythia.core.reinforcement.e_greedy_policies import NormalEpsilonGreedyPolicy
from pythia.tests.doubles import QFunctionWrapper

STATE_A = [0]
STATE_B = [1]

@pytest.fixture
def q_function():
    return QFunctionWrapper([0, 1])


@pytest.fixture
def zero_policy():
    """Returns a normal e greedy policy with epsilon 0"""
    return NormalEpsilonGreedyPolicy(0)


@pytest.fixture
def epsilon_policy():
    """Returns a normal e greedy policy with epsilon 5"""
    return NormalEpsilonGreedyPolicy(5)


@pytest.fixture
def function_policy():
    """Returns a normal e greedy policy with epsilon 5 provided by a function"""
    return NormalEpsilonGreedyPolicy(lambda: 5)


def test_epsilon_zero(zero_policy, q_function):
    q_function.set_state_action_values(STATE_A, -1, 1)

    assert 1 == zero_policy.select(STATE_A, q_function)


def test_multiple_states(zero_policy, q_function):
    q_function.set_state_action_values(STATE_A, -1, 1)
    q_function.set_state_action_values(STATE_B, 10, -5)

    assert 1 == zero_policy.select(STATE_A, q_function)
    assert 0 == zero_policy.select(STATE_B, q_function)


def test_non_zero_epsilon(epsilon_policy, q_function):
    np.random.seed(7)

    q_function.set_state_action_values(STATE_A, -1, 1)

    assert 0 == epsilon_policy.select(STATE_A, q_function)


def test_epsilon_as_function(function_policy, q_function):
    np.random.seed(7)

    q_function.set_state_action_values(STATE_A, -1, 1)

    assert 0 == function_policy.select(STATE_A, q_function)


def test_incomplete_state(zero_policy, q_function):
    q_function[STATE_A, 0] = -1
    assert 1 == zero_policy.select(STATE_A, q_function)