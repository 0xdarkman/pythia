from abc import abstractmethod, ABC
from collections import namedtuple

import numpy as np

from pythia.core.agents.cnn_ensamble import CNNEnsemble
from pythia.core.agents.fpm_agent import FpmAgent
from pythia.core.agents.random_agent import RuleAgent


class FpmRunner(ABC):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    @property
    def window(self):
        return self.config["training"]["window"]

    @property
    def coins(self):
        return self.config["trading"]["coins"]

    @property
    def agent(self):
        d = self.config["setup"]["agent"]
        return namedtuple("Agent", d.keys())(**d)

    def _make_agent(self, tf_sess):
        self.logger.info("Using agent: {}".format(self.agent))
        if self.agent.type == "FpmAgent":
            ann = CNNEnsemble(tf_sess, len(self.coins), self.window, self.config)
            return FpmAgent(ann, self._get_memory(), self._make_random_portfolio, self.config, self.logger)
        elif self.agent.type == "RandomAgent":
            return RuleAgent(self._make_random_portfolio)
        elif self.agent.type == "HoldAgent":
            return RuleAgent(self.make_fix_portfolio)
        else:
            raise UnknownConfiguration("The requested agent '{}' is unknown.".format(self.agent))

    @abstractmethod
    def _get_memory(self):
        pass

    def make_fix_portfolio(self):
        zeros = np.zeros(len(self.coins) + 1)
        zeros[1:] = self.agent.holding
        return self._normalize_tensor(zeros)

    def _make_random_portfolio(self):
        return self._normalize_tensor(np.random.random_sample((len(self.coins) + 1,)))

    @staticmethod
    def _normalize_tensor(tensor):
        norm = np.linalg.norm(tensor, ord=1)
        if norm == 0:
            norm = np.finfo(tensor.dtype).eps
        return tensor / norm


class UnknownConfiguration(AttributeError):
    pass
