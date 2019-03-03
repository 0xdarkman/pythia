import argparse
import json
import os
from datetime import datetime

# TODO: Remove
import pandas as pd
import pytz
import tensorflow as tf

from pythia.core.agents.fpm_memory import FPMMemory
from pythia.core.environment.fpm_environment import FpmEnvironment
from pythia.core.fpm_runner import FpmRunner
from pythia.core.sessions.fpm_session import FpmSession
from pythia.core.streams.fpm_time_series import FpmTimeSeries


class FpmService(FpmRunner):
    def __init__(self, config, log_fn):
        super(FpmService, self).__init__(config)
        self.tf_saver = tf.train.Saver()
        self.config = config
        self.log = log_fn

        # TODO: Remove
        self.data_directory = R"/home/bernhard/repos/pythia/data/recordings/poloniex/processed"

    @property  # TODO: Remove
    def cash(self):
        return self.config["trading"]["cash"]

    @property
    def model(self):
        return self.config["setup"]["model_path"]

    @property
    def memory(self):
        return self.config["setup"]["memory_path"]

    def run(self):
        self._check_file_paths()
        with tf.Session() as sess:
            agent = self._make_agent(sess)
            sess.run(tf.global_variables_initializer())
            agent.restore(self.model)
            self._run_testing(agent)

    def _check_file_paths(self):
        if not os.path.exists(os.path.dirname(self.model)):
            msg = "The model specified does not exist: {}".format(self.model)
            self.log.error(msg)
            raise self.LoadingError(msg)
        if not os.path.exists(self.memory):
            msg = "The memory specified does not exist: {}".format(self.memory)
            self.log.error(msg)
            raise self.LoadingError(msg)

    def _get_memory(self):
        return FPMMemory(self.config)

    def _run_testing(self, agent):
        fpm_sess = self._make_session(self.config["testing"], agent)
        reward = fpm_sess.run()
        self._log_reward(reward)
        self.log.info("Finished testing with final reward of {}".format(reward))
        return reward

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
        s = FpmSession(env, agent, self._log_reward, None)
        s.log_interval = 1000
        return s

    def _log_reward(self, reward):
        self.log.info("Intermediate reward {:.4f}".format(reward))

    class LoadingError(FileNotFoundError):
        pass


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.log_stream = None

    def info(self, msg):
        self.out("[{}][INFO] {}".format(datetime.now(tz=pytz.utc), msg))

    def error(self, msg):
        self.out("[{}][ERROR] {}".format(datetime.now(tz=pytz.utc), msg))

    def __enter__(self):
        if self.log_file is not None:
            self.log_stream = open(self.log_file, 'a+')
            self.out = self.file_writer
        else:
            self.out = print
        return self

    def file_writer(self, msg):
        self.log_stream.write(msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_stream is not None:
            self.log_stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pythia online on Poloniex")
    parser.add_argument("--config", dest="config",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "fpm_online.json"),
                        help="Path to the config file used")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    with Logger(cfg["log"].get("log_file", None)) as log:
        service = FpmService(cfg, log)
        service.run()
