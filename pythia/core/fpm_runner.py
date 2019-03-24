from abc import abstractmethod, ABC

import numpy as np

from pythia.core.agents.cnn_ensamble import CNNEnsemble
from pythia.core.agents.fpm_agent import FpmAgent


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

    def _make_agent(self, tf_sess):
        ann = CNNEnsemble(tf_sess, len(self.coins), self.window, self.config)
        return FpmAgent(ann, self._get_memory(), self._make_random_portfolio, self.config, self.logger)

    @abstractmethod
    def _get_memory(self):
        pass

    def _make_random_portfolio(self):
        return self._normalize_tensor(np.random.random_sample((len(self.coins) + 1,)))

    @staticmethod
    def _normalize_tensor(tensor):
        norm = np.linalg.norm(tensor, ord=1)
        if norm == 0:
            norm = np.finfo(tensor.dtype).eps
        return tensor / norm
