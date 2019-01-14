import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import Summary

from pythia.core.agents.cnn_ensamble import CNNEnsemble
from pythia.core.agents.fpm_agent import FpmAgent
from pythia.core.agents.fpm_memory import FPMMemory
from pythia.core.environment.fpm_environment import FpmEnvironment
from pythia.core.sessions.fpm_session import FpmSession
from pythia.core.streams.fpm_time_series import FpmTimeSeries
from pythia.core.streams.poloniex_history import PoloniexHistory


class FpmBackTest:
    def __init__(self, config, data_directory):
        self.config = config
        self.data_directory = data_directory
        self.tf_saver = tf.train.Saver()
        self.log = print

        self._time_series = None
        self._tf_board_writer = None

    @property
    def episodes(self):
        return self.config["training"]["episodes"]

    @property
    def window(self):
        return self.config["training"]["window"]

    @property
    def coins(self):
        return self.config["trading"]["coins"]

    @property
    def cash(self):
        return self.config["trading"]["cash"]

    def run(self, output_directory):
        if self._should_update_to_latest():
            self._update_to_latest()
        self._time_series = self._load_time_series()

        self._tf_board_writer = self._make_tf_board_writer(output_directory)
        ckpt_path = os.path.join(output_directory, "model.ckpt")
        with tf.Session() as sess:
            if tf.train.checkpoint_exists(ckpt_path):
                self.log("[INFO] restoring agent from: {}".format(ckpt_path))
                self.tf_saver.restore(sess, ckpt_path)

            agent = self._make_agent(sess)
            fpm_sess = self._make_session_for_agent(agent)
            sess.run(tf.global_variables_initializer())
            for i in range(self.episodes):
                r = fpm_sess.run()
                self.log("[INFO] Episode {}: saving agent to: {}".format(i, ckpt_path))
                self.tf_saver.save(sess, ckpt_path)
                self.log("[PERFORMANCE] Last reward of episode {}: {}".format(i, r))
                self._log_reward(r)
                self._tf_board_writer.flush()

    def _should_update_to_latest(self):
        return self.config["trading"]["update_to_latest"]

    def _update_to_latest(self):
        self.log("[INFO] Updating Poloniex Historical data...")
        h = PoloniexHistory(self.config, self.data_directory)
        h.update()

    def _load_time_series(self):
        rates_dir = os.path.join(self.data_directory, "processed")
        data_frames = []
        for coin in self.coins:
            with open(os.path.join(rates_dir, "{}_{}.csv".format(self.cash, coin))) as r:
                data_frames.append(pd.read_csv(r))

        return FpmTimeSeries(*data_frames)

    @staticmethod
    def _make_tf_board_writer(out_dir):
        return tf.summary.FileWriter(os.path.join(out_dir, "log"))

    def _make_agent(self, tf_sess):
        ann = CNNEnsemble(tf_sess, len(self.coins), self.window, self.config)
        mem = FPMMemory(self.config)
        return FpmAgent(ann, mem, self._make_random_portfolio, self.config)

    def _make_random_portfolio(self):
        return self._normalize_tensor(np.random.random_sample((len(self.coins) + 1,)))

    @staticmethod
    def _normalize_tensor(tensor):
        norm = np.linalg.norm(tensor, ord=1)
        if norm == 0:
            norm = np.finfo(tensor.dtype).eps
        return tensor / norm

    def _make_session_for_agent(self, agent):
        env = FpmEnvironment(self._time_series, self.config)
        s = FpmSession(env, agent, self._log_reward)
        s.log_interval = 1000
        return s

    def _log_reward(self, reward):
        self.log("[PERFORMANCE] Intermediate reward {}".format(reward))
        stat_reward = Summary(value=[Summary.Value(tag="reward", simple_value=reward)])
        self._tf_board_writer.add_summary(stat_reward)


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "fpm_default.json"), "r") as f:
        cfg = json.load(f)

    backTest = FpmBackTest(cfg, R"/home/bernhard/repos/pythia/data/recordings/poloniex")
    backTest.run(R"/home/bernhard/repos/pythia/data/models/fpm")
