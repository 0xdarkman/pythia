import inspect
import os

import pytest

from pythia.core.environment.crypto_ai_environment import RatesAiEnvironment
from pythia.core.environment.crypto_rewards import TotalBalanceReward
from pythia.core.environment.rigged_policy import STOP_AT_THRESHOLD, RiggedPolicy
from pythia.core.streams.shape_shift_rates import ShapeShiftRates

COIN_A = "BTC"
COIN_B = "ETH"


class PolicyDummy:
    def select(self, s, q):
        return 0


class QDummy:
    pass


@pytest.fixture
def env():
    dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    path = os.path.join(dir, "test_data/2018-02-28-shapeshift-BTC_ETH.json".format(COIN_A, COIN_B))
    with open(path) as stream:
        rates = ShapeShiftRates(stream, preload=True)
        yield RatesAiEnvironment(rates, COIN_A, "10", 1, {1: COIN_A, 2: COIN_B}, TotalBalanceReward())


@pytest.fixture
def q_function():
    return QDummy()


def test_rigged_policy_produces_profit(env, q_function):
    policy = RiggedPolicy(env, PolicyDummy(), 1.0, rigging_distance=STOP_AT_THRESHOLD, threshold=0.03)
    s = env.reset()
    start_balance = env.balance_in(COIN_A)
    done = False
    while not done:
        a = policy.select(s, q_function)
        s, _, done, _ = env.step(a)

    assert (env.balance_in(COIN_A) - start_balance) > 0
