import pandas as pd
import tensorflow as tf

from pythia.core.agents.dqt_agent import DQTAgent, DQTDiffStateTransformer, DQTRewardCalc, DQTRatioStateTransformer
from pythia.core.environment.exchange_trading_environment import ExchangeTradingEnvironment
from pythia.core.models.dqt_model import DQTModel
from pythia.core.sessions.rates_exchange_session import RatesExchangeSession
from pythia.core.streams.data_frame_stream import DataFrameStream, CASH_TOKEN
from pythia.core.utils.profiling import clock_block
from pythia.core.visualization.coin_exchange_visualizer import CoinExchangeVisualizer

TOKEN = "TOKEN"
TRAIN_EPISODES = 10
lr = 1e-6
tau = 3e-4
token = pd.read_csv("../data/recordings/30min/BTC_USD.csv", index_col=0)
train = token[:"2018-05-15"]
test = token["2018-05-16":]


def build_model():
    return DQTModel(lr=lr, input_size=200, hidden_layers=[128, 128, 64])


with tf.Session():
    exchange = "{}_{}".format(CASH_TOKEN, TOKEN)
    rates = DataFrameStream(train, name=TOKEN)
    vis = CoinExchangeVisualizer(rates)
    env_train = ExchangeTradingEnvironment(rates, CASH_TOKEN, start_amount=1000, window=201,
                                           state_transform=DQTRatioStateTransformer(exchange),
                                           reward_calculator=DQTRewardCalc(100, exchange))
    env_train.register_listener(vis.record_exchange)
    agent = DQTAgent(build_model, [CASH_TOKEN, None, TOKEN], 1, 0.85, 64, 64, tau)
    sess_train = RatesExchangeSession(env_train, agent)

    for e in range(TRAIN_EPISODES):
        with clock_block("Running"):
            sess_train.run()
        print("Training episode {} finished.".format(e))
        print("Token difference after training: {0}".format(sess_train.difference()))
        #vis.render(exchange)

    env_test = ExchangeTradingEnvironment(DataFrameStream(test, name=TOKEN), CASH_TOKEN, start_amount=1000, window=201,
                                          state_transform=DQTRatioStateTransformer(exchange),
                                          reward_calculator=DQTRewardCalc(100, exchange))
    sess_test = RatesExchangeSession(env_test, agent)
    with clock_block("Testing"):
        sess_test.run()
    print("Token difference after test run: {0}".format(sess_test.difference()))
