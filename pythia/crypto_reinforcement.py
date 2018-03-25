import sys

import tensorflow as tf

from pythia.core.agents.td_agent import TDAgent
from pythia.core.environment.crypto_ai_environment import CryptoAiEnvironment, ActionFilter
from pythia.core.environment.crypto_rewards import TotalBalanceReward
from pythia.core.reinforcement.e_greedy_policies import NormalEpsilonGreedyPolicy, EpsilonGreedyPolicy
from pythia.core.reinforcement.q_neuronal import QNeuronal
from pythia.core.reinforcement.q_regression_model import QRegressionModel
from pythia.core.sessions.crypto_exchange_session import CryptoExchangeSession
from pythia.core.streams.shape_shift_rates import ShapeShiftRates
from pythia.core.utils.profiling import clock_block
from pythia.core.visualization.coin_exchange_visualizer import CoinExchangeVisualizer

COIN_A = "RLC"
COIN_B = "WINGS"
LEARNING_RATE = 0.01
MEMORY_SIZE = 10
ALPHA = 0.2
GAMMA = 0.9
WINDOW = 10
START_EPS = 1
TOTAL_EPISODES = 1000
n = 10

if __name__ == '__main__':
    path = "../data/recordings/filtered/2018-02-28-shapeshift-{}_{}.json".format(COIN_A, COIN_B) if len(sys.argv) == 1 else sys.argv[0]
    with open(path) as stream, tf.Session():
        with clock_block("Initialization"):
            rates = ShapeShiftRates(stream, preload=True)
            vis = CoinExchangeVisualizer(rates)
            env = CryptoAiEnvironment(rates, COIN_A, "0.1", WINDOW, {1: COIN_A, 2: COIN_B}, TotalBalanceReward())
            env.register_listener(vis.record_exchange)

            model = QRegressionModel(3 + WINDOW * 2, [100], LEARNING_RATE)
            Q = QNeuronal(model, MEMORY_SIZE)
            episode = 0
            agent = TDAgent(EpsilonGreedyPolicy(0.1, ActionFilter(env)), Q, n, GAMMA, ALPHA)
            sess = CryptoExchangeSession(env, agent)

        for e in range(TOTAL_EPISODES):
            episode = e
            with clock_block("Running"):
                sess.run()
            print("Episode {} finished.".format(episode))
            print("The td agent crated a coin difference of: {0}".format(sess.difference()))

        print("Current balance: {0} {1}".format(env.amount, env.coin))
        print("Exchange actions: {0}".format(vis.actions))

        rates.reset()
        vis.render("BTC_ETH")
