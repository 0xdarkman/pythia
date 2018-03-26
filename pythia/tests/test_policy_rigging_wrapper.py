from collections import deque

import pytest

from pythia.core.environment.crypto_ai_environment import _make_coin_to_index
from pythia.tests.crypto_doubles import entry, RecordsStub, RatesStub


class RiggedPolicy:
    def __init__(self, env, inner_policy, rigging_chance):
        self.env = env
        self.inner_policy = inner_policy
        self.rigging_chance = rigging_chance
        self.rigged_actions = deque()

    def select(self, state, q_function):
        if self.rigging_chance == 1.0 and len(self.rigged_actions) == 0:
            self.rigged_actions = self._make_rigged_actions()

        if len(self.rigged_actions) != 0:
            return self.rigged_actions.popleft()
        return self.inner_policy.select(state, q_function)

    def _make_rigged_actions(self):
        targets = list(filter(lambda c: c != self.env.coin, self.env.action_to_coin.values()))
        pairs = next(self.env.rates_stream, None)
        mins = dict()
        maxs = dict()
        coin = self.env.coin
        step = 0
        while pairs is not None:
            for target in targets:
                rate = pairs[coin + "_" + target].rate
                if target not in mins or rate < mins[target][0]:
                    mins[target] = (rate, step)
                if target not in maxs or rate > maxs[target][0]:
                    maxs[target] = (rate, step)
            pairs = next(self.env.rates_stream, None)
            step += 1

        if len(mins) == 0 or len(maxs) == 0:
            return deque()

        best_target = None
        biggest_diff = float("-inf")
        for target in targets:
            if maxs[target][1] > mins[target][1]:
                continue

            diff = maxs[target][0] - mins[target][0]
            if diff > biggest_diff:
                biggest_diff = diff
                best_target = target

        if best_target is None:
            return deque()

        exchange_to = self._coin_to_action(best_target)
        exchange_back = self._coin_to_action(coin)
        return deque([0] * maxs[best_target][1] + [exchange_to] +
                     [0] * (mins[best_target][1] - maxs[best_target][1] - 1) + [exchange_back])

    def _coin_to_action(self, coin):
        for k, v in self.env.action_to_coin.items():
            if v == coin:
                return k


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


def make_policy(env, inner_policy, chance):
    return RiggedPolicy(env, inner_policy, chance)


def test_zero_chance_of_rigging_just_invokes_inner_policy(policy_spy, q_function):
    policy = make_policy(EnvironmentDummy(), policy_spy, 0.0)
    policy.select(A_STATE, q_function)
    assert policy_spy.received_select_args == ([0], q_function)


def test_not_rigged_returns_action_from_inner_policy(q_function):
    policy = make_policy(EnvironmentDummy(), PolicyStub(1), 0.0)
    assert policy.select(A_STATE, q_function) == 1


def test_rigging_looks_ahead_for_profitable_exchange_and_selects_actions_accordingly(policy_spy, q_function):
    env = CryptoEnvironmentStub(action_to_coin={1: "BTC", 2: "ETH"}, active_coin="BTC")
    env.add_record(entry("BTC_ETH", "2")). \
        add_record(entry("BTC_ETH", "4")). \
        add_record(entry("BTC_ETH", "1")). \
        add_record(entry("BTC_ETH", "2")).finish()
    policy = make_policy(env, policy_spy, 1.0)
    assert policy.select(A_STATE, q_function) == 0
    assert policy.select(A_STATE, q_function) == 2
    assert policy.select(A_STATE, q_function) == 1
    policy.select(OTHER_STATE, q_function)
    assert policy_spy.received_select_args == (OTHER_STATE, q_function)
