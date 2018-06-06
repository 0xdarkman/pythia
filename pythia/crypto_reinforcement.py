import sys

import tensorflow as tf

from reinforcement.agents.td_agent import TDAgent
from pythia.core.environment.rates_ai_environment import RatesAiEnvironment, ActionFilter
from pythia.core.environment.rates_rewards import TotalBalanceReward
from reinforcement.policies.e_greedy_policies import NormalEpsilonGreedyPolicy
from pythia.core.environment.rigged_policy import STOP_AT_THRESHOLD, RiggedPolicy
from reinforcement.reward_functions.q_neuronal import QNeuronal
from reinforcement.models.q_regression_model import QRegressionModel
from pythia.core.sessions.rates_exchange_session import RatesExchangeSession
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
            env = RatesAiEnvironment(rates, COIN_A, "10", WINDOW, {1: COIN_A, 2: COIN_B}, TotalBalanceReward())
            env.register_listener(vis.record_exchange)

            model = QRegressionModel(3 + WINDOW * 2, [100], LEARNING_RATE)
            Q = QNeuronal(model, MEMORY_SIZE)
            episode = 0
            policy = RiggedPolicy(env,
                                  NormalEpsilonGreedyPolicy(lambda: START_EPS / (episode + 1), ActionFilter(env)),
                                  0.5, rigging_distance=STOP_AT_THRESHOLD, threshold=0.5)
            agent = TDAgent(policy, Q, n, GAMMA, ALPHA)
            sess = RatesExchangeSession(env, agent)

        for e in range(TOTAL_EPISODES):
            episode = e
            with clock_block("Running"):
                sess.run()
            print("Episode {} finished.".format(episode))
            print("The td agent crated a token difference of: {0}".format(sess.difference()))


        print("Current balance: {0} {1}".format(env.amount, env.token))
        print("Exchange actions: {0}".format(vis.actions))
        rates.reset()
        vis.render("BTC_ETH")
