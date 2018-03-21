import random

import pytest

from pythia.core.agents.td_agent import TDAgent
from pythia.core.environment.crypto_ai_environment import CryptoAiEnvironment
from pythia.core.environment.crypto_rewards import TotalBalanceReward
from pythia.core.reinforcement.e_greedy_policies import EpsilonGreedyPolicy, NormalEpsilonGreedyPolicy
from pythia.core.reinforcement.q_ann import QAnn
from pythia.core.reinforcement.q_regression_model import QRegressionModel
from pythia.core.reinforcement.q_table import QTable
from pythia.core.sessions.crypto_exchange_session import CryptoExchangeSession
from pythia.tests.crypto_doubles import RecordsStub, RatesStub, entry


@pytest.fixture
def env():
    rate = RatesStub(RecordsStub())
    rate.add_record(entry("BTC_ETH", "2.0", miner_fee="0"), entry("ETH_BTC", "0.5", miner_fee="0")) \
        .add_record(entry("BTC_ETH", "2.0", miner_fee="0"), entry("ETH_BTC", "0.5", miner_fee="0")) \
        .add_record(entry("BTC_ETH", "4.0", miner_fee="0"), entry("ETH_BTC", "0.25", miner_fee="0")) \
        .add_record(entry("BTC_ETH", "2.0", miner_fee="0"), entry("ETH_BTC", "0.5", miner_fee="0")) \
        .add_record(entry("BTC_ETH", "1.0", miner_fee="0"), entry("ETH_BTC", "1.0", miner_fee="0")) \
        .add_record(entry("BTC_ETH", "2.0", miner_fee="0"), entry("ETH_BTC", "0.5", miner_fee="0")) \
        .add_record(entry("BTC_ETH", "4.0", miner_fee="0"), entry("ETH_BTC", "0.25", miner_fee="0")).finish()
    yield CryptoAiEnvironment(rate, "BTC", "1", 1, {1: "BTC", 2: "ETH"}, TotalBalanceReward())
    rate.close()


def test_td_agent_produces_sensible_q_values(env):
    q_table = QTable([0, 1, 2])
    agent = TDAgent(EpsilonGreedyPolicy(0.2), q_table, 1, 0.9, 0.1)
    sess = CryptoExchangeSession(env, agent)
    for _ in range(0, 100):
        sess.run()

    assert q_table[[0, 0.0, 1.0, 0.0], 2] > q_table[[0, 0.0, 1.0, 0.0], 0]
    assert q_table[[0, 0.0, 1.0, 0.0], 2] > q_table[[0, 0.0, 1.0, 0.0], 1]


def test_td_agent_produces_sensible_regression_model_predictions(env):
    model = QRegressionModel(5, [100], 0.1)
    Q = QAnn(model, [0, 1, 2], 10)
    episode = 0
    agent = TDAgent(NormalEpsilonGreedyPolicy(lambda: (1 / (episode + 1))), Q, 1, 0.9, 0.1)
    sess = CryptoExchangeSession(env, agent)
    for e in range(0, 100):
        episode = e
        sess.run()

    assert Q[[0, 0.0, 1.0, 0.0], 2] > Q[[0, 0.0, 1.0, 0.0], 0]
    assert Q[[0, 0.0, 1.0, 0.0], 2] > Q[[0, 0.0, 1.0, 0.0], 1]