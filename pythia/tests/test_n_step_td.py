from collections import deque

import pytest

from reinforcement.policies.e_greedy_policies import EpsilonGreedyPolicy
from pythia.core.n_step_sarsa import NStepSarsa
from reinforcement.reward_functions.q_table import QTable


class EnvironmentSim(object):
    def __init__(self):
        self.records = deque()
        self.current_records = deque()

    def record_states(self, *state_rewards):
        for sr in state_rewards:
            self.record(sr[0], sr[1])

    def record(self, state, reward):
        self.records.append((state, reward))

    def reset(self):
        self.current_records = deque(self.records)
        return self.current_records.popleft()[0]

    def step(self, action):
        s, r = self.current_records.popleft()
        return s, r, len(self.current_records) == 0, None


class EnvironmentSpy(object):
    def __init__(self, episode_length):
        self.received_actions = []
        self.episode_length = episode_length

    def reset(self):
        return 0

    def step(self, action):
        self.received_actions.append(action)
        self.episode_length -= 1
        return 0, 0, self.episode_length == 0, None


class PolicyStub(object):
    def __init__(self, *selected_actions):
        self.selected_actions = deque(selected_actions)

    def select(self, state, q_func):
        return self.selected_actions.popleft()


@pytest.fixture
def environment_sim():
    """Returns a simulated environment used by the TD algorithm"""
    return EnvironmentSim()


@pytest.fixture
def environment_spy():
    """Returns a spying environment recording the actions performed in it."""
    return EnvironmentSpy(4)


def make_n_step_td(env, action_space, gamma=1.0, alpha=1.0, steps=1):
    q_table = QTable(action_space)
    policy = EpsilonGreedyPolicy(0)
    tdn = NStepSarsa(env, q_table, policy, action_space)
    tdn.steps = steps
    tdn.gamma = gamma
    tdn.alpha = alpha

    return tdn


def make_n_step_td_with_policy(env, policy, action_space):
    q_table = QTable(action_space)
    tdn = NStepSarsa(env, q_table, policy, action_space)
    tdn.steps = 1
    tdn.gamma = 1.0
    tdn.alpha = 1.0

    return tdn


def test_td_zero_one_action_one_state(environment_sim):
    tdn = make_n_step_td(environment_sim, [0])
    environment_sim.record_states((0, 0), (1, 2))

    tdn.run()

    assert tdn.q_function[0, 0] == 2


def test_multiple_states_no_back_propagation(environment_sim):
    tdn = make_n_step_td(environment_sim, [0], 0)
    environment_sim.record_states((0, 0), (1, 2), (2, -1))

    tdn.run()

    assert tdn.q_function[0, 0] == 2
    assert tdn.q_function[1, 0] == -1


def test_back_propagation_first_step(environment_sim):
    tdn = make_n_step_td(environment_sim, [0], 0.5, 0.5)
    environment_sim.record_states((0, 0), (1, 2), (2, -1))

    tdn.run()

    assert tdn.q_function[0, 0] == 1
    assert tdn.q_function[1, 0] == -0.5


def test_back_propagation_second_step(environment_sim):
    tdn = make_n_step_td(environment_sim, [0], 0.5, 0.5)
    environment_sim.record_states((0, 0), (1, 2), (2, -1))

    tdn.run()
    tdn.run()

    assert tdn.q_function[0, 0] == 1.375
    assert tdn.q_function[1, 0] == -0.75


def test_actions_follow_policy(environment_spy):
    tdn = make_n_step_td_with_policy(environment_spy, PolicyStub(1, 1, 0, 1), [0, 1])

    tdn.run()

    assert environment_spy.received_actions == [1, 1, 0, 1]


def test_correct_action_values_are_updated(environment_sim):
    tdn = make_n_step_td_with_policy(environment_sim, PolicyStub(0, 1), [0, 1])
    environment_sim.record_states((0, 0), (1, 2), (2, 3))

    tdn.run()

    assert 2 == tdn.q_function[0, 0]
    assert 3 == tdn.q_function[1, 1]


def test_steps_greater_one(environment_sim):
    tdn = make_n_step_td(environment_sim, [0], 1.0, 1.0, 2)
    environment_sim.record_states((0, 0), (1, 2), (2, -1), (3, 0), (4, 4))

    tdn.run()

    assert 1 == tdn.q_function[0, 0]
    assert -1 == tdn.q_function[1, 0]
    assert 4 == tdn.q_function[2, 0]
    assert 4 == tdn.q_function[3, 0]


def test_gamma_falloff(environment_sim):
    tdn = make_n_step_td(environment_sim, [0], 0.5, 1.0, 3)
    tdn.q_function[3, 0] = 2
    environment_sim.record_states((0, 0), (1, 2), (2, -1), (3, 3), (4, 4))

    tdn.run()

    assert 2.5 == tdn.q_function[0, 0]
    assert 1.5 == tdn.q_function[1, 0]
    assert 5 == tdn.q_function[2, 0]
    assert 4 == tdn.q_function[3, 0]