import random

import pytest

from pythia.core.reinforcement.e_greedy_policies import EpsilonGreedyPolicy
from pythia.core.reinforcement.tests.common_doubles import Call, MockFilter
from pythia.core.reinforcement.tests.q_doubles import QFunctionWrapper


def make_policy(epsilon, filter=None):
    return EpsilonGreedyPolicy(epsilon, filter)


STATE_A = [0]
STATE_B = [1]


@pytest.fixture
def q_function():
    return QFunctionWrapper([0, 1])


@pytest.fixture
def zero_policy():
    """Returns a normal e greedy policy with epsilon 0"""
    return make_policy(0)


@pytest.fixture
def epsilon_policy():
    """Returns a normal e greedy policy with epsilon 0.2"""
    return make_policy(0.2)


@pytest.fixture
def function_policy():
    """Returns a normal e greedy policy with epsilon 0.2 provided by a function"""
    return make_policy(lambda: 0.2)


def test_epsilon_zero(zero_policy, q_function):
    q_function.set_state_action_values(STATE_A, -1, 1)

    assert zero_policy.select(STATE_A, q_function) == 1


def test_multiple_states(zero_policy, q_function):
    q_function.set_state_action_values(STATE_A, -1, 1)
    q_function.set_state_action_values(STATE_B, 10, -5)

    assert zero_policy.select(STATE_A, q_function) == 1
    assert zero_policy.select(STATE_B, q_function) == 0


def test_non_zero_epsilon(epsilon_policy, q_function):
    random.seed(1)

    q_function.set_state_action_values(STATE_A, -1, 1)

    assert epsilon_policy.select(STATE_A, q_function) == 0


def test_epsilon_as_function(function_policy, q_function):
    random.seed(1)

    q_function.set_state_action_values(STATE_A, -1, 1)

    assert function_policy.select(STATE_A, q_function) == 0


def test_incomplete_state(zero_policy, q_function):
    q_function[STATE_A, 0] = -1

    assert zero_policy.select(STATE_A, q_function) == 1


def test_invalid_actions_are_ignored(q_function):
    q_function[STATE_A, 0] = 10
    q_function[STATE_A, 1] = -1
    filter = MockFilter(Call((STATE_A, 0), returns=False), Call((STATE_A, 1), returns=True))
    policy = make_policy(0, filter)
    assert policy.select(STATE_A, q_function) == 1
