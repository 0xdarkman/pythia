import random

import pytest

from pythia.core.reinforcement.e_greedy_policies import RiggedPolicy
from pythia.tests.crypto_doubles import entry, RecordsStub, RatesStub


class CryptoEnvironmentStub:
    def __init__(self, action_to_coin, active_coin):
        self.rates_stream = RatesStub(RecordsStub())
        self.action_to_coin = action_to_coin
        self.coin = active_coin

    def add_record(self, *entries):
        self.rates_stream.add_record(*entries)
        return self

    def finish(self):
        self.rates_stream.finish()

    def step(self, action):
        next(self.rates_stream)


class EnvironmentDummy:
    pass


class PolicySpy:
    def __init__(self):
        self.received_select_args = None

    def select(self, state, q_function):
        self.received_select_args = (state, q_function)


class PolicyStub:
    def __init__(self, action):
        self.a = action

    def select(self, state, q_function):
        return self.a


class PolicyDummy:
    pass


class QDummy:
    pass


A_STATE = [0]
OTHER_STATE = [1]


@pytest.fixture
def policy_spy():
    return PolicySpy()


@pytest.fixture
def q_function():
    return QDummy()


def make_policy(env, inner_policy, chance, distance=None):
    return RiggedPolicy(env, inner_policy, chance, distance)


def test_zero_chance_of_rigging_just_invokes_inner_policy(policy_spy, q_function):
    policy = make_policy(EnvironmentDummy(), policy_spy, 0.0)
    policy.select(A_STATE, q_function)
    assert policy_spy.received_select_args == ([0], q_function)


def test_not_rigged_returns_action_from_inner_policy(q_function):
    policy = make_policy(EnvironmentDummy(), PolicyStub(1), 0.0)
    assert policy.select(A_STATE, q_function) == 1


def test_rigging_looks_ahead_for_profitable_exchange_and_selects_actions_accordingly(q_function):
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "2")).finish()
    policy = make_policy(env, PolicyDummy(), 1.0)
    assert policy.select(A_STATE, q_function) == 0
    assert policy.select(A_STATE, q_function) == 2
    assert policy.select(A_STATE, q_function) == 1


def test_rigging_looks_ahead_only_specified_distance(q_function):
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "3")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "6")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "2")).finish()
    policy = make_policy(env, PolicyDummy(), 1.0, 3)
    assert policy.select(A_STATE, q_function) == 0
    assert policy.select(A_STATE, q_function) == 2
    assert policy.select(A_STATE, q_function) == 1


def test_rigging_starts_at_current_rate_position(q_function):
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "2")).finish()
    env.step(0)
    policy = make_policy(env, policy_spy, 1.0)
    assert policy.select(A_STATE, q_function) == 2
    assert policy.select(A_STATE, q_function) == 1


def test_rigging_leaves_rates_unaffected(q_function):
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "3")).finish()
    policy = make_policy(env, policy_spy, 1.0)
    policy.select(A_STATE, q_function)
    assert next(env.rates_stream)["BTC_ETH"] == entry("BTC_ETH", "2")


def test_after_rigging_continue_with_inner_policy(policy_spy, q_function):
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "2")).finish()
    policy = make_policy(env, policy_spy, 1.0)
    env.step(policy.select(A_STATE, q_function))
    env.step(policy.select(A_STATE, q_function))
    env.step(policy.select(A_STATE, q_function))
    policy.select(OTHER_STATE, q_function)
    assert policy_spy.received_select_args == (OTHER_STATE, q_function)


def test_rigging_happens_by_specified_change(q_function):
    random.seed(7)
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "2")).finish()
    policy = make_policy(env, PolicyDummy(), 0.5)
    assert policy.select(A_STATE, q_function) == 0


def test_count_number_of_riggings(q_function):
    random.seed(7)
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "2")).finish()
    policy = make_policy(env, PolicyStub(0), 0.2)
    env.step(policy.select(A_STATE, q_function))
    env.step(policy.select(A_STATE, q_function))
    env.step(policy.select(A_STATE, q_function))
    env.step(policy.select(A_STATE, q_function))
    assert policy.rigging_count == 1


def test_return_maximum_difference_exchange_actions(q_function):
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "3")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "5")). \
        add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "6")). \
        add_record(entry("BTC_ETH", "4")).finish()
    policy = make_policy(env, PolicyDummy(), 1.0)
    assert policy.select(A_STATE, q_function) == 0
    assert policy.select(A_STATE, q_function) == 0
    assert policy.select(A_STATE, q_function) == 2
    assert policy.select(A_STATE, q_function) == 1


@pytest.mark.skip
def test_return_good_exchange_in_complex_environment(q_function):
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "3")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "6")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "5")). \
        add_record(entry("BTC_ETH", "3")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "7")). \
        add_record(entry("BTC_ETH", "3")).finish()

