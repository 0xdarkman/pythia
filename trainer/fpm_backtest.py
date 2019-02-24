import json
import os
import random
import time

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


def _fix_seed():
    tf.set_random_seed(42)
    np.random.seed(42)
    random.seed(42)


class FpmBackTest:
    def __init__(self, config, data_directory):
        self.config = config
        self.data_directory = data_directory
        self.tf_saver = tf.train.Saver()
        self.log = print
        self.show_profiling = self.config["log"].get("profiling", False)
        if self.config["setup"].get("fix_seed", False):
            _fix_seed()

        self._tf_board_writer = None
        self._last_intermediate = None
        self._assets_log = list()

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

    @property
    def update_to_latest(self):
        return self.config["setup"]["update_to_latest"]

    @property
    def restore(self):
        return self.config["setup"]["restore_last_checkpoint"]

    def run(self, output_directory):
        if self.update_to_latest:
            self._update_to_latest()

        self._tf_board_writer = self._make_tf_board_writer(output_directory)
        ckpt_path = os.path.join(output_directory, "model.ckpt")
        with tf.Session() as sess:
            if self.restore and tf.train.checkpoint_exists(ckpt_path):
                self.log("[INFO] restoring agent from: {}".format(ckpt_path))
                self.tf_saver.restore(sess, ckpt_path)

            agent = self._make_agent(sess)
            sess.run(tf.global_variables_initializer())
            self._run_training(agent, ckpt_path, output_directory, sess)
            self._run_testing(agent)

    @staticmethod
    def _make_tf_board_writer(out_dir):
        return tf.summary.FileWriter(os.path.join(out_dir, "log"))

    def _update_to_latest(self):
        self.log("[INFO] Updating Poloniex Historical data...")
        h = PoloniexHistory(self.config, self.data_directory)
        h.update()

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

    def _run_training(self, agent, ckpt_path, output_directory, sess):
        fpm_sess = self._make_session(self.config["training"], agent)
        reward = 0
        for i in range(self.episodes):
            reward = fpm_sess.run()
            self.log("[INFO] Episode {}: saving agent to: {}".format(i, ckpt_path))
            self.tf_saver.save(sess, ckpt_path)
            self.log("[PERFORMANCE] Last reward of episode {}: {:.4f}".format(i, reward))
            self._log_reward(reward)
            self._tf_board_writer.flush()
            if self.config["setup"]["record_assets"]:
                np.save(os.path.join(output_directory, "assets.gz"))

        self.log("[INFO] Finished training with final reward of {}".format(reward))

    def _make_session(self, series_cfg, agent):
        series = self._load_time_series(series_cfg)
        fpm_sess = self._make_session_for_agent(agent, series)
        return fpm_sess

    def _load_time_series(self, config):
        rates_dir = os.path.join(self.data_directory, "processed")
        data_frames = []
        for coin in self.coins:
            with open(os.path.join(rates_dir, "{}_{}.csv".format(self.cash, coin))) as r:
                df = pd.read_csv(r).set_index('timestamp')
                data_frames.append(df[config["start"]:config["end"]])

        return FpmTimeSeries(*data_frames)

    def _make_session_for_agent(self, agent, series):
        env = FpmEnvironment(series, self.config)
        recorder = self._record_assets if self.config["setup"]["record_assets"] else None
        s = FpmSession(env, agent, self._log_reward, recorder)
        s.log_interval = 1000
        return s

    def _run_testing(self, agent):
        fpm_sess = self._make_session(self.config["training"], agent)
        reward = fpm_sess.run()
        self._log_reward(reward)
        self._tf_board_writer.flush()
        self.log("[INFO] Finished testing with final reward of {}".format(reward))

    def _log_reward(self, reward):
        if self.show_profiling:
            self._log_performance()
        self.log("[PERFORMANCE] Intermediate reward {:.4f}".format(reward))
        stat_reward = Summary(value=[Summary.Value(tag="reward", simple_value=reward)])
        self._tf_board_writer.add_summary(stat_reward)

    def _log_performance(self):
        now = time.perf_counter()
        if self._last_intermediate is not None:
            delta = now - self._last_intermediate
            self.log("[PROFILE] Intermediate calculations took {:.4f}".format(delta))
        self._last_intermediate = now

    def _record_assets(self, env):
        self._assets_log.append(np.array(env.assets))


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "fpm_default.json"), "r") as f:
        cfg = json.load(f)

    backTest = FpmBackTest(cfg, R"/home/bernhard/repos/pythia/data/recordings/poloniex")
    backTest.run(R"/home/bernhard/repos/pythia/data/models/fpm")
