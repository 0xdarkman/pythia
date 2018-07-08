import os
import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary
from reinforcement.agents.td_agent import TDAgent
from reinforcement.models.q_regression_model import QRegressionModel
from reinforcement.policies.e_greedy_policies import NormalEpsilonGreedyPolicy
from reinforcement.reward_functions.q_neuronal import QNeuronal
from tensorflow.python.lib.io import file_io

from pythia.core.environment.rates_ai_environment import ExchangeTradingAiEnvironment, ActionFilter
from pythia.core.environment.rates_rewards import TotalBalanceReward
from pythia.core.sessions.rates_exchange_session import RatesExchangeSession
from pythia.core.streams.share_rates import ShareRates, Symbol
from pythia.core.utils.profiling import clock_block

EVAL_STEPS = 10


def run_shares_dqn_regression_model(holding_tokens,
                                    buying_tokens,
                                    starting_balance,
                                    window,
                                    hidden_layers,
                                    learning_rate,
                                    memory_size,
                                    epsilon_episode_start,
                                    num_steps,
                                    gamma,
                                    alpha,
                                    episodes,
                                    output_dir=None):
    token_h = "CURRENCY" if holding_tokens is None else os.path.basename(holding_tokens)
    token_b = os.path.basename(buying_tokens)

    open_fn = open
    if buying_tokens.startswith("gs://"):
        open_fn = lambda name: file_io.FileIO(name, 'r')

    with open_fn(buying_tokens) as stream, tf.Session() as sess:
        with clock_block("Initialization"):
            rates = ShareRates(Symbol(token_b, stream))
            env = ExchangeTradingAiEnvironment(rates, token_h, starting_balance, window, {1: token_h, 2: token_b},
                                               TotalBalanceReward())

            model = QRegressionModel(3 + window * 2, hidden_layers, learning_rate)
            saver, ckpt = None, None
            if output_dir is not None:
                saver, ckpt = tf.train.Saver(), os.path.join(output_dir, "model.ckpt")
            if ckpt is not None and tf.train.checkpoint_exists(ckpt):
                saver.restore(sess, ckpt)

            Q = QNeuronal(model, n=3, memory_size=memory_size)
            episode = 0
            policy = NormalEpsilonGreedyPolicy(lambda: epsilon_episode_start / (episode + 1), ActionFilter(env))
            agent = TDAgent(policy, Q, num_steps, gamma, alpha)
            rates_sess = RatesExchangeSession(env, agent)

        for e in range(episodes):
            episode = e
            with clock_block("Running"):
                rates_sess.run()
            print("Episode {} finished.".format(episode))
            print("The td agent crated a token difference of: {0}".format(rates_sess.difference()))
            if output_dir is not None:
                saver.save(sess, ckpt)

        with clock_block("Evaluation"):
            diff = 0.0
            for _ in range(EVAL_STEPS):
                rates_sess.run()
                diff += float(rates_sess.difference())
            effectiveness = diff / float(EVAL_STEPS)
            if output_dir is not None:
                summary = Summary(value=[Summary.Value(tag="effectiveness", simple_value=effectiveness)])
                eval_path = os.path.join(output_dir, "effectiveness")
                summary_writer = tf.summary.FileWriter(eval_path)
                summary_writer.add_summary(summary)
                summary_writer.flush()

        print("Current balance: {0} {1}".format(env.amount, env.token))
        print("Effectiveness: {0}".format(effectiveness))


if __name__ == '__main__':
    run_shares_dqn_regression_model(
        holding_tokens=None,
        buying_tokens="../data/recordings/shares/SPY.csv",
        starting_balance=1000,
        window=10,
        hidden_layers=[100],
        learning_rate=0.001,
        memory_size=10,
        epsilon_episode_start=1,
        gamma=0.9,
        alpha=0.2,
        num_steps=10,
        episodes=300,
        output_dir="../output"
    )
