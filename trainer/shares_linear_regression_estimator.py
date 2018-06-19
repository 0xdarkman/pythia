import os
import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary
from reinforcement.agents.td_agent import TDAgent
from reinforcement.models.q_regression_model import QRegressionModel
from reinforcement.policies.e_greedy_policies import NormalEpsilonGreedyPolicy
from reinforcement.reward_functions.q_neuronal import QNeuronal
from tensorflow.python.lib.io import file_io

from pythia.core.environment.rates_ai_environment import RatesAiEnvironment, ActionFilter
from pythia.core.environment.rates_rewards import TotalBalanceReward
from pythia.core.models.q_linear_regression_model import QLinearRegressionModel
from pythia.core.sessions.rates_exchange_session import RatesExchangeSession
from pythia.core.streams.share_rates import ShareRates, Symbol
from pythia.core.utils.profiling import clock_block

EVAL_STEPS = 10


def run_shares_linear_regression_estimator(holding_tokens,
                                           buying_tokens,
                                           starting_balance,
                                           window,
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
            env = RatesAiEnvironment(rates, token_h, starting_balance, window, {1: token_h, 2: token_b},
                                     TotalBalanceReward())

            model = QLinearRegressionModel(3 + window * 2, lr=learning_rate)
            Q = QNeuronal(model, n=3, memory_size=memory_size)
            episode = 0
            policy = NormalEpsilonGreedyPolicy(lambda: epsilon_episode_start / (episode + 1), ActionFilter(env))
            agent = TDAgent(policy, Q, num_steps, gamma, alpha)
            rates_sess = RatesExchangeSession(env, agent)

        diff_writer, saver, ckpt_path = None,  None, None
        if output_dir:
            diff_path = os.path.join(output_dir, "difference")
            diff_writer = tf.summary.FileWriter(diff_path)
            saver, ckpt_path = tf.train.Saver(), os.path.join(output_dir, "model.ckpt")
            if tf.train.checkpoint_exists(ckpt_path):
                saver.restore(sess, ckpt_path)

        for e in range(episodes):
            episode = e
            with clock_block("Running"):
                rates_sess.run()
            print("Episode {} finished.".format(episode))
            difference = rates_sess.difference()
            print("The td agent crated a token difference of: {0}".format(difference))
            if diff_writer:
                stat_diff = Summary(value=[Summary.Value(tag="difference", simple_value=float(difference))])
                diff_writer.add_summary(stat_diff)
            if saver:
                saver.save(sess, ckpt_path)

        if diff_writer is not None:
            diff_writer.flush()

        with clock_block("Evaluation"):
            diff = 0.0
            for _ in range(EVAL_STEPS):
                rates_sess.run()
                diff += float(rates_sess.difference())
            effectiveness = diff / float(EVAL_STEPS)
            if output_dir is not None:
                stat_effectiveness = Summary(value=[Summary.Value(tag="effectiveness", simple_value=effectiveness)])
                effectiveness_path = os.path.join(output_dir, "effectiveness")
                effectiveness_writer = tf.summary.FileWriter(effectiveness_path)
                effectiveness_writer.add_summary(stat_effectiveness)
                effectiveness_writer.flush()

        print("Current balance: {0} {1}".format(env.amount, env.token))
        print("Effectiveness: {0}".format(effectiveness))


def main():
    run_shares_linear_regression_estimator(
        holding_tokens=None,
        buying_tokens="../data/recordings/shares/daily/SPY.csv",
        starting_balance=1000,
        window=30,
        learning_rate=0.001,
        memory_size=20,
        epsilon_episode_start=0.1,
        gamma=0.9,
        alpha=0.2,
        num_steps=15,
        episodes=1000,
        output_dir="../output"
    )


if __name__ == '__main__':
    main()
