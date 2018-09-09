from collections import deque

import pytest

from pythia.core.sessions.fpm_session import FpmSession


class StateStub:
    def __init__(self, idx):
        self.index = idx

    def __eq__(self, other):
        return self.index == other.index

    def __repr__(self):
        return f"StateStub({self.index})"


class ActionStub:
    def __init__(self, idx):
        self.index = idx

    def __eq__(self, other):
        return self.index == other.index

    def __repr__(self):
        return f"ActionStub({self.index})"


class EnvironmentStub:
    def __init__(self):
        self.states = None
        self.rewards = None
        self.set_states(*range(1, 100))
        self.set_rewards(*range(1, 100))

    def set_states(self, *state_indices):
        self.states = deque(StateStub(i) for i in state_indices)

    def set_rewards(self, *rewards):
        self.rewards = deque(rewards)

    def reset(self):
        return self.states.popleft()

    def step(self, action):
        s = self.states.popleft()
        return s, self.rewards.popleft(), len(self.states) == 0, None


class EnvironmentSpy(EnvironmentStub):
    def __init__(self):
        super().__init__()
        self.has_been_reset = False
        self.received_actions = list()

    def reset(self):
        self.has_been_reset = True
        return super().reset()

    def step(self, action):
        self.received_actions.append(action)
        return super().step(action)


class AgentStub:
    def __init__(self):
        self.actions = None
        self.set_actions(*range(1, 100))

    def set_actions(self, *action_indices):
        self.actions = deque(ActionStub(i) for i in action_indices)

    def step(self, state):
        return self.actions.popleft()


class AgentSpy(AgentStub):
    def __init__(self):
        super().__init__()
        self.received_states = list()

    def step(self, state):
        self.received_states.append(state)
        return super().step(state)


class LoggerSpy:
    def __init__(self):
        self.received_rewards = list()

    def __call__(self, r):
        self.received_rewards.append(r)


@pytest.fixture
def env():
    return EnvironmentSpy()


@pytest.fixture
def agent():
    return AgentSpy()


@pytest.fixture
def logger():
    return LoggerSpy()


@pytest.fixture
def sess(env, agent, logger):
    return FpmSession(env, agent, logger)


def make_state(index):
    return StateStub(index)


def make_states(*indices):
    return [make_state(i) for i in indices]


def make_action(index):
    return ActionStub(index)


def make_actions(*indices):
    return [make_action(i) for i in indices]


def make_rewards(*rewards):
    return list(rewards)


def test_resets_the_environment_when_run(sess, env):
    assert not env.has_been_reset
    sess.run()
    assert env.has_been_reset


def test_agent_receives_reset_state(sess, env, agent):
    env.set_states(1, 2)
    sess.run()
    assert agent.received_states[0] == make_state(1)


def test_environment_receives_action_from_agent(sess, env, agent):
    env.set_states(1, 2)
    agent.set_actions(1, 2)
    sess.run()
    assert env.received_actions[0] == make_action(1)


def test_agent_receives_following_states_of_environment(sess, env, agent):
    env.set_states(1, 2, 3)
    sess.run()
    assert agent.received_states == make_states(1, 2, 3)


def test_environment_receives_actions_from_agent(sess, env, agent):
    env.set_states(1, 2, 3)
    agent.set_actions(1, 2, 3)
    sess.run()
    assert env.received_actions == make_actions(1, 2)


def test_sessions_logs_reward_from_environment(sess, env, logger):
    env.set_states(1, 2)
    env.set_rewards(10)
    sess.run()
    assert logger.received_rewards == make_rewards(10)


def test_session_logs_reward_with_specified_interval(sess, env, logger):
    env.set_rewards(*range(1, 100))
    sess.log_interval = 10
    sess.run()
    assert logger.received_rewards == make_rewards(*range(1, 100, 10))


def test_a_log_interval_of_zero_means_no_logging(sess, logger):
    sess.log_interval = 0
    sess.run()
    assert len(logger.received_rewards) == 0


def test_session_run_returns_last_reward(sess, env):
    env.set_states(1, 2, 3)
    env.set_rewards(10, 20)
    assert sess.run() == 20
