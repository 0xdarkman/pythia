import sys

from pythia.core.agents.td_agent import TDAgent
from pythia.core.environment.crypto_ai_environment import CryptoAiEnvironment
from pythia.core.environment.crypto_rewards import RatesChangeReward
from pythia.core.reinforcement.e_greedy_policies import NormalEpsilonGreedyPolicy
from pythia.core.reinforcement.q_ann import QAnn
from pythia.core.reinforcement.q_regression_model import QRegressionModel
from pythia.core.sessions.crypto_exchange_session import CryptoExchangeSession
from pythia.core.streams.shape_shift_rates import ShapeShiftRates
from pythia.core.utils.profiling import clock_block
from pythia.core.visualization.coin_exchange_visualizer import CoinExchangeVisualizer

COIN_A = "BTC"
COIN_B = "ETH"
LEARNING_RATE = 0.01
MEMORY_SIZE = 10
ALPHA = 0.8
GAMMA = 0.99
WINDOW = 1
START_EPS = 100
TOTAL_EPISODES = 10000

model = QRegressionModel(3 + WINDOW * 2, [100], LEARNING_RATE)
Q = QAnn(model, [0, 1, 2], MEMORY_SIZE)
episode = 0
agent = TDAgent(NormalEpsilonGreedyPolicy(lambda: (START_EPS / (episode + 1))), Q, WINDOW, GAMMA, ALPHA)

if __name__ == '__main__':
    path = "../data/recordings/filtered/2018-02-28-shapeshift-BTC_ETH.json" if len(sys.argv) == 1 else sys.argv[0]
    with open(path) as stream:
        with clock_block("Initialization"):
            rates = ShapeShiftRates(stream, preload=True)
            vis = CoinExchangeVisualizer(rates)
            env = CryptoAiEnvironment(rates, COIN_A, "0.1", 1, {1: COIN_A, 2: COIN_B}, RatesChangeReward())
            env.register_listener(vis.record_exchange)
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
