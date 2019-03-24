import json
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import Summary

from pythia.core.agents.fpm_memory import FPMMemory
from pythia.core.environment.fpm_environment import FpmEnvironment
from pythia.core.fpm_runner import FpmRunner
from pythia.core.sessions.fpm_session import FpmSession
from pythia.core.streams.fpm_time_series import FpmTimeSeries
from pythia.core.streams.poloniex_history import PoloniexHistory
from pythia.logger import Logger


def _set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class FpmBackTest(FpmRunner):
    def __init__(self, config, data_directory, log_fn):
        super(FpmBackTest, self).__init__(config, log_fn)
        self.data_directory = data_directory
        self.show_profiling = self.config["log"].get("profiling", False)
        self.seed = self.config["setup"].get("fixed_seed", np.random.randint(2 ** 32 - 1))
        _set_seed(self.seed)
        self.logger.info("Seed used is: {}".format(self.seed))

        self._tf_board_writer = None
        self._last_intermediate = None
        self._assets_log = list()

    @property
    def episodes(self):
        return self.config["training"]["episodes"]

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
        with tf.Session() as sess:
            agent = self._make_agent(sess)
            if self.restore and tf.train.checkpoint_exists(os.path.join(output_directory, agent.model_file_name)):
                agent.restore(output_directory)

            sess.run(tf.global_variables_initializer())
            self._run_training(agent, output_directory)
            return self._run_testing(agent)

    def _get_memory(self):
        return FPMMemory(self.config)

    @staticmethod
    def _make_tf_board_writer(out_dir):
        return tf.summary.FileWriter(os.path.join(out_dir, "log"))

    def _update_to_latest(self):
        self.logger.info("Updating Poloniex Historical data...")
        h = PoloniexHistory(self.config, self.data_directory)
        h.update()

    def _run_training(self, agent, output_directory):
        fpm_sess = self._make_session(self.config["training"], agent)
        reward = 0
        for i in range(self.episodes):
            reward = fpm_sess.run()
            fpm_sess.agent.save(output_directory)
            self.logger.info("Last reward of episode {}: {:.4f}".format(i, reward))
            self._log_reward(reward)
            self._tf_board_writer.flush()
            if self.config["setup"]["record_assets"]:
                np.save(os.path.join(output_directory, "assets.gz"))

        self.logger.info("Finished training with final reward of {}".format(reward))

    def _make_session(self, series_cfg, agent):
        series = self._load_time_series(series_cfg)
        fpm_sess = self._make_session_for_agent(agent, series)
        return fpm_sess

    def _load_time_series(self, config):
        data_frames = []
        for coin in self.coins:
            with open(os.path.join(self.data_directory, "{}_{}.csv".format(self.cash, coin))) as r:
                df = pd.read_csv(r, index_col='timestamp')
                df.index = pd.to_datetime(df.index, unit='s')
                data_frames.append(df[config["start"]:config.get("end", None)])

        return FpmTimeSeries(*data_frames)

    def _make_session_for_agent(self, agent, series):
        env = FpmEnvironment(series, self.config)
        recorder = self._record_assets if self.config["setup"]["record_assets"] else None
        s = FpmSession(env, agent, self._log_reward, recorder)
        s.log_interval = 1000
        return s

    def _run_testing(self, agent):
        fpm_sess = self._make_session(self.config["testing"], agent)
        reward = fpm_sess.run()
        self._log_reward(reward)
        self._tf_board_writer.flush()
        self.logger.info("Finished testing with final reward of {}".format(reward))
        return reward

    def _log_reward(self, reward):
        if self.show_profiling:
            self._log_performance()
        self.logger.info("Intermediate reward {:.4f}".format(reward))
        stat_reward = Summary(value=[Summary.Value(tag="reward", simple_value=reward)])
        self._tf_board_writer.add_summary(stat_reward)

    def _log_performance(self):
        now = time.perf_counter()
        if self._last_intermediate is not None:
            delta = now - self._last_intermediate
            self.logger.info("Intermediate calculations took {:.4f}".format(delta))
        self._last_intermediate = now

    def _record_assets(self, env):
        self._assets_log.append(np.array(env.assets))


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "fpm_default.json"), "r") as f:
        cfg = json.load(f)

    with Logger() as logger:
        backTest = FpmBackTest(cfg, R"/home/bernhard/repos/pythia/data/recordings/poloniex/processed", logger)
        r = backTest.run(R"/home/bernhard/repos/pythia/data/models/fpm")
        logger.info("[FINISH] New reward: {}".format(r))
        logger.info("[FINISH] Saving seed {}".format(backTest.seed))
        with open(R"/home/bernhard/repos/pythia/data/seed.txt", 'a+') as f:
            f.write("{},{}\n".format(backTest.seed, r))
