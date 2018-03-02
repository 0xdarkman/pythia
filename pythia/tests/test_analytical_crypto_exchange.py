from collections import deque

import pytest

from pythia.environment.crypto_environment import CryptoEnvironment
from pythia.tests.doubles import RatesStub, entry, RecordsStub


class CryptoExchangeSession:
    def __init__(self, env, agent):
        self.environment = env
        self.agent = agent

    def run(self):
        s = self.environment.reset()
        done = False
        while not done:
            a = self.agent.step(s)
            s, _, done, _ = self.environment.step(a)


class EnvironmentStub(CryptoEnvironment):
    def __init__(self):
        self.rates = RatesStub(RecordsStub())

    def add_record(self, *pairs):
        self.rates.add_record(*pairs)
        return self

    def finish(self):
        self.rates.finish()
        super().__init__(self.rates, "BTC", 2)

    def close(self):
        self.rates.close()


class EnvironmentSpy(EnvironmentStub):
    def __init__(self):
        super().__init__()
        self.add_record(entry("BTC_ETH", 1))\
            .add_record(entry("BTC_ETH", 2))\
            .add_record(entry("BTC_ETH", 3))\
            .finish()
        self.received_actions = list()

    def step(self, action):
        self.received_actions.append(action)
        return super().step(action)


class AgentSpy:
    def __init__(self):
        self.received_states = list()

    def step(self, state):
        self.received_states.append(state)


class AgentStub:
    def __init__(self):
        self.actions = deque()

    def set_actions(self, *actions):
        self.actions = deque(actions)

    def step(self, state):
        return self.actions.popleft()


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


def test_agent_receives_all_states_but_last(env, agent_spy):
    env.add_record(entry("BTC_ETH", 1)).add_record(entry("BTC_ETH", 2)).add_record(entry("BTC_ETH", 3)).finish()
    CryptoExchangeSession(env, agent_spy).run()
    assert agent_spy.received_states == [{"BTC_ETH": entry("BTC_ETH", 1)}, {"BTC_ETH": entry("BTC_ETH", 2)}]


def test_environment_receives_agent_actions(env_spy, agent):
    agent.set_actions(None, "ETH")
    CryptoExchangeSession(env_spy, agent).run()
    assert env_spy.received_actions == [None, "ETH"]


@pytest.mark.skip("Needs to be implemented")
def test_calculating_profit():
    pass


@pytest.mark.skip("Needs to be implemented")
def test_calculating_loss():
    pass
