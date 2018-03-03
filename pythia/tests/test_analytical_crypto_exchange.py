from collections import deque

import pytest
from decimal import Decimal

from pythia.environment.crypto_environment import CryptoEnvironment
from pythia.sessions.crypto_exchange_session import CryptoExchangeSession
from pythia.tests.doubles import RatesStub, entry, RecordsStub


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
        self.add_record(entry("BTC_ETH", 1)) \
            .add_record(entry("BTC_ETH", 2)) \
            .add_record(entry("BTC_ETH", 3)) \
            .finish()
        self.received_actions = list()

    def step(self, action):
        self.received_actions.append(action)
        return super().step(action)


class AgentDoubleBase:
    def __init__(self):
        self.start_coin = "BTC"

    def balance_in(self, coin):
        return Decimal('0')


class AgentSpy(AgentDoubleBase):
    def __init__(self):
        super().__init__()
        self.received_states = list()

    def step(self, state):
        self.received_states.append(state)


class AgentStub(AgentDoubleBase):
    def __init__(self):
        super().__init__()
        self.actions = deque()
        self.balance = None

    def set_actions(self, *actions):
        self.actions = deque(actions)

    def set_balance(self, balance):
        self.balance = balance

    def step(self, state):
        return self.actions.popleft()

    def balance_in(self, coin):
        return self.balance


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


def test_agent_receives_all_states_but_last(env, agent_spy):
    env.add_record(entry("BTC_ETH", 1)).add_record(entry("BTC_ETH", 2)).add_record(entry("BTC_ETH", 3)).finish()
    CryptoExchangeSession(env, agent_spy).run()
    assert agent_spy.received_states == [{"BTC_ETH": entry("BTC_ETH", 1)}, {"BTC_ETH": entry("BTC_ETH", 2)}]


def test_environment_receives_agent_actions(env_spy, agent):
    agent.set_actions(None, "ETH")
    CryptoExchangeSession(env_spy, agent).run()
    assert env_spy.received_actions == [None, "ETH"]


def test_calculating_profit(simple_env, agent):
    agent.set_actions(None)
    agent.set_balance(Decimal('2'))
    sess = CryptoExchangeSession(simple_env, agent)
    sess.run()
    agent.set_balance(Decimal('4'))
    assert sess.difference() == Decimal('2')


def test_calculating_loss(simple_env, agent):
    agent.set_actions(None)
    agent.set_balance(Decimal('2'))
    sess = CryptoExchangeSession(simple_env, agent)
    sess.run()
    agent.set_balance(Decimal('1'))
    assert sess.difference() == Decimal('-1')
