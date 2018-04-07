import pytest
import tensorflow as tf

from reinforcement.agents.td_agent import TDAgent
from reinforcement.models.q_regression_model import QRegressionModel
from reinforcement.policies.e_greedy_policies import EpsilonGreedyPolicy, NormalEpsilonGreedyPolicy
from reinforcement.reward_functions.q_neuronal import QNeuronal
from reinforcement.reward_functions.q_table import QTable

from pythia.core.environment.crypto_ai_environment import CryptoAiEnvironment, ActionFilter
from pythia.core.environment.crypto_rewards import TotalBalanceReward
from pythia.core.sessions.crypto_exchange_session import CryptoExchangeSession
from pythia.tests.crypto_doubles import RecordsStub, RatesStub, entry


@pytest.fixture(scope="session", autouse=True)
def config_tensorflow():
    original_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(3)
    tf.set_random_seed(7)
    yield
    tf.logging.set_verbosity(original_v)


@pytest.fixture(autouse=True)
def set_session():
    tf.reset_default_graph()
    with tf.Session():
        yield


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
    Q = QNeuronal(model, 3, 10)
    episode = 0
    agent = TDAgent(NormalEpsilonGreedyPolicy(lambda: (1 / (episode + 1))), Q, 1, 0.9, 0.1)
    sess = CryptoExchangeSession(env, agent)
    for e in range(0, 100):
        episode = e
        sess.run()

    assert Q[[0, 0.0, 1.0, 0.0], 2] > Q[[0, 0.0, 1.0, 0.0], 0]
    assert Q[[0, 0.0, 1.0, 0.0], 2] > Q[[0, 0.0, 1.0, 0.0], 1]


def test_td_agent_does_not_perform_invalid_actions_when_filtered(env):
    q_table = QTable([0, 1, 2], initializer=lambda: 0)
    agent = TDAgent(EpsilonGreedyPolicy(0.2, ActionFilter(env)), q_table, 1, 0.9, 0.1)
    sess = CryptoExchangeSession(env, agent)
    for _ in range(0, 100):
        sess.run()

    assert q_table[[0, 0.0, 1.0, 0.0], 1] == 0

