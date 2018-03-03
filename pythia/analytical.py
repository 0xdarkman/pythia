import sys

from pythia.core.agents.analytical_agent import AnalyticalAgent
from pythia.core.environment.crypto_environment import CryptoEnvironment
from pythia.core.sessions.crypto_exchange_session import CryptoExchangeSession
from pythia.core.streams.shape_shift_rates import ShapeShiftRates

agent = AnalyticalAgent('0.06', '0.001', 2)

if __name__ == '__main__':
    path = "../data/recordings/2018-02-28-shapeshift-exchange-records.json" if len(sys.argv) == 1 else sys.argv[0]
    with open(path) as stream:
        rates = ShapeShiftRates(stream)
        env = CryptoEnvironment(rates, "BTC", 10)
        sess = CryptoExchangeSession(env, agent)
        sess.run()

        print("The analytical agent crated a coin difference of: {0}".format(sess.difference()))
        print("Current balance: {0} {1}".format(env.amount, env.coin))