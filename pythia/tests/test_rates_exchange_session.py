from collections import deque

import pytest
from decimal import Decimal

from pythia.core.environment.rates_environment import RatesEnvironment
from pythia.core.sessions.rates_exchange_session import RatesExchangeSession
from pythia.tests.crypto_doubles import RatesStub, entry, RecordsStub


class EnvironmentStub(RatesEnvironment):
    def __init__(self):
        self.rates = RatesStub(RecordsStub())
        self.rewards = deque()

    def add_record(self, pairs, reward=0):
        if isinstance(pairs, tuple):
            self.rates.add_record(*pairs)
        else:
            self.rates.add_record(pairs)
        self.rewards.append(reward)
        return self

    def step(self, action):
        s, _, done, _ = super().step(action)
        return s, self.rewards.popleft(), done, None

    def finish(self):
        self.rewards.popleft()  # starting state has no reward
        self.rates.finish()
        super().__init__(self.rates, "BTC", 2, 1)

    def close(self):
        self.rates.close()


class EnvironmentSpy(EnvironmentStub):
    def __init__(self):
        super().__init__()
        self.add_record((entry("BTC_ETH", 1), entry("ETH_BTC", 1))) \
            .add_record((entry("BTC_ETH", 2), entry("ETH_BTC", "0.5"))) \
            .add_record((entry("BTC_ETH", 3), entry("ETH_BTC", "0.3"))) \
            .finish()
        self.received_actions = list()

    def step(self, action):
        self.received_actions.append(action)
        return super().step(action)


class AgentSpy:
    def __init__(self):
        self.received_start_state = None
        self.received_finish_reward = None
        self.received_states = list()
        self.received_rewards = list()

    def start(self, state):
        self.received_start_state = state
        return None

    def step(self, state, reward):
        self.received_states.append(state)
        self.received_rewards.append(reward)
        return None

    def finish(self, reward):
        self.received_finish_reward = reward


class AgentStub:
    def __init__(self):
        self.actions = deque()

    def set_actions(self, *actions):
        self.actions = deque(actions)

    def start(self, state):
        return self.actions.popleft()

    def step(self, state, reward):
        return self.actions.popleft()

    def finish(self, reward):
        pass


@pytest.fixture
def env():
    e = EnvironmentStub()
    yield e
    e.close()


@pytest.fixture
def agent_spy():
    return AgentSpy()


@pytest.fixture
def env_spy():
    e = EnvironmentSpy()
    yield e
    e.close()


@pytest.fixture
def agent():
    return AgentStub()


@pytest.fixture
def simple_env(env):
    env.add_record(entry("BTC_ETH", 1)).add_record(entry("BTC_ETH", 2)).finish()
    yield env
    env.close()


def make_session(env, agent):
    return RatesExchangeSession(env, agent)


def test_agent_received_initial_state_as_start(env, agent_spy):
    env.add_record(entry("BTC_ETH", 1)).add_record(entry("BTC_ETH", 2)).finish()
    make_session(env, agent_spy).run()
    assert agent_spy.received_start_state == {"token": "BTC", "balance": Decimal("2"),
                                              "rates": [{"BTC_ETH": entry("BTC_ETH", 1)}]}


def test_agent_step_receives_all_states_but_first_and_last(env, agent_spy):
    env.add_record(entry("BTC_ETH", 1)) \
        .add_record(entry("BTC_ETH", 2)) \
        .add_record(entry("BTC_ETH", 3)) \
        .add_record(entry("BTC_ETH", 4)).finish()
    make_session(env, agent_spy).run()
    assert agent_spy.received_states == [
        {"token": "BTC", "balance": Decimal("2"), "rates": [{"BTC_ETH": entry("BTC_ETH", 2)}]},
        {"token": "BTC", "balance": Decimal("2"), "rates": [{"BTC_ETH": entry("BTC_ETH", 3)}]}]


def test_agent_step_received_all_but_last_reward(env, agent_spy):
    env.add_record(entry("BTC_ETH", 1), None) \
        .add_record(entry("BTC_ETH", 2), 3) \
        .add_record(entry("BTC_ETH", 3), -10) \
        .add_record(entry("BTC_ETH", 4), 9).finish()
    make_session(env, agent_spy).run()
    assert agent_spy.received_rewards == [3, -10]


def test_agent_finish_receives_last_reward(env, agent_spy):
    env.add_record(entry("BTC_ETH", 1), None).add_record(entry("BTC_ETH", 2), 10).finish()
    make_session(env, agent_spy).run()
    assert agent_spy.received_finish_reward == 10


def test_environment_receives_agent_actions(env_spy, agent):
    agent.set_actions(None, "ETH")
    make_session(env_spy, agent).run()
    assert env_spy.received_actions == [None, "ETH"]


def test_calculating_profit(env, agent):
    env.add_record(entry("BTC_ETH", 1)) \
        .add_record(entry("BTC_ETH", 2, miner_fee=0)) \
        .add_record(entry("ETH_BTC", 1)).finish()
    agent.set_actions(None, "ETH")
    sess = make_session(env, agent)
    sess.run()
    assert sess.difference() == Decimal(2)


def test_calculating_loss(env, agent):
    env.add_record(entry("BTC_ETH", 1)) \
        .add_record(entry("BTC_ETH", 2, miner_fee=0)) \
        .add_record(entry("ETH_BTC", "0.5")).finish()
    agent.set_actions(None, "ETH")
    sess = make_session(env, agent)
    sess.run()
    assert sess.difference() == Decimal(0)