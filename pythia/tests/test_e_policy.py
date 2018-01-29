import random

import pytest

from pythia.reinforcement.e_greedy_policies import EpsilonGreedyPolicy
from pythia.reinforcement.q_table import QTable


class QFunctionWrapper(QTable):
    def set_state_action_values(self, state, zero_value, one_value):
        self[state, 0] = zero_value
        self[state, 1] = one_value


def make_policy(qf, epsilon):
    return EpsilonGreedyPolicy(qf, epsilon)


STATE_A = [0]
STATE_B = [1]


@pytest.fixture
def q_function():
    """A Q-Table with a 2 dimensional action space [0, 1]"""
    return QFunctionWrapper([0, 1])


def test_epsilon_zero(q_function):
    policy = make_policy(q_function, 0)

    q_function.set_state_action_values(STATE_A, -1, 1)

    assert policy.select(STATE_A) == 1


def test_multiple_states(q_function):
    policy = make_policy(q_function, 0)

    q_function.set_state_action_values(STATE_A, -1, 1)
    q_function.set_state_action_values(STATE_B, 10, -5)

    assert policy.select(STATE_A) == 1
    assert policy.select(STATE_B) == 0


def test_non_zero_epsilon(q_function):
    policy = make_policy(q_function, 0.2)
    random.seed(1)

    q_function.set_state_action_values(STATE_A, -1, 1)

    assert policy.select(STATE_A) == 0


def test_epsilon_as_function(q_function):
    policy = make_policy(q_function, lambda: 0.2)
    random.seed(1)

    q_function.set_state_action_values(STATE_A, -1, 1)

    assert policy.select(STATE_A) == 0


def test_incomplete_state(q_function):
    policy = make_policy(q_function, 0)

    q_function[STATE_A, 0] = -1

    assert policy.select(STATE_A) == 1
