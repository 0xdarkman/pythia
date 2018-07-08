import sys

from pythia.core.agents.analytical_agent import AnalyticalAgent
from pythia.core.environment.exchange_trading_environment import ExchangeTradingEnvironment
from pythia.core.sessions.rates_exchange_session import RatesExchangeSession
from pythia.core.streams.shape_shift_rates import ShapeShiftRates, SUPPORTED_COINS
from pythia.core.utils.profiling import clock_block
from pythia.core.visualization.coin_exchange_visualizer import CoinExchangeVisualizer

targets = list(SUPPORTED_COINS)
targets.remove("BTC")
agent = AnalyticalAgent('0.1', '0', 2, targets)

if __name__ == '__main__':
    path = "../data/recordings/2018-02-28-shapeshift-exchange-records.json" if len(sys.argv) == 1 else sys.argv[0]
    with open(path) as stream:
        with clock_block("Initialization"):
            rates = ShapeShiftRates(stream, preload=True)
            vis = CoinExchangeVisualizer(rates)
            env = ExchangeTradingEnvironment(rates, "BTC", "0.1")
            env.register_listener(vis.record_exchange)
            sess = RatesExchangeSession(env, agent)

        with clock_block("Running"):
            sess.run()

        print("The analytical agent crated a token difference of: {0}".format(sess.difference()))
        print("Current balance: {0} {1}".format(env.amount, env.token))
        print("Exchange actions: {0}".format(vis.actions))

        rates.reset()
        vis.render("BTC_GAME")