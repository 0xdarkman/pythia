import os

import tensorflow as tf
from reinforcement.agents.td_agent import TDAgent
from reinforcement.models.q_regression_model import QRegressionModel
from reinforcement.policies.e_greedy_policies import NormalEpsilonGreedyPolicy
from reinforcement.reward_functions.q_neuronal import QNeuronal

from pythia.core.environment.rates_ai_environment import RatesAiEnvironment, ActionFilter
from pythia.core.environment.rates_rewards import TotalBalanceReward
from pythia.core.sessions.rates_exchange_session import RatesExchangeSession
from pythia.core.streams.share_rates import ShareRates, Symbol
from pythia.core.utils.profiling import clock_block


def run_shares_model(holding_tokens,
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

    with open(buying_tokens) as stream, tf.Session() as sess:
        with clock_block("Initialization"):
            rates = ShareRates(Symbol(token_b, stream))
            env = RatesAiEnvironment(rates, token_h, starting_balance, window, {1: token_h, 2: token_b},
                                     TotalBalanceReward())

            model = QRegressionModel(3 + window * 2, hidden_layers, learning_rate, 42)
            saver, ckpt = None, None
            if output_dir is not None:
                saver, ckpt = tf.train.Saver(), os.path.join(output_dir, "model.ckpt")
            if ckpt is not None and tf.train.checkpoint_exists(ckpt):
                saver.restore(sess, ckpt)

            Q = QNeuronal(model, memory_size)
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

        print("Current balance: {0} {1}".format(env.amount, env.token))


if __name__ == '__main__':
    run_shares_model(
        holding_tokens=None,
        buying_tokens="../data/recordings/shares/SPY.csv",
        starting_balance=10,
        window=10,
        hidden_layers=[100],
        learning_rate=0.01,
        memory_size=10,
        epsilon_episode_start=1,
        gamma=0.9,
        alpha=0.2,
        num_steps=10,
        episodes=300,
        output_dir="../output"
    )
