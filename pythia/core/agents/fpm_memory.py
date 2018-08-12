import itertools
from collections.__init__ import deque

import numpy as np


class FPMMemory:
    def __init__(self, window, size, beta):
        self._window = window
        self.beta = beta

        self._prices = deque(maxlen=size)
        self._portfolio = None
        self._num_assets = None

    def record(self, prices, portfolio):
        self._validate_input(len(prices), portfolio)
        self._prices.append(prices)
        self._portfolio = np.array(portfolio)

    def _validate_input(self, m, portfolio):
        if m + 1 != len(portfolio):
            raise self.DataMismatchError("Amount of asset symbols and length of portfolio vector does not match: "
                                         "m={} len(portfolio)={}", m, len(portfolio))
        if self._num_assets is None:
            self._num_assets = m

    def _make_price_tensor(self, start):
        p = np.array(list(itertools.islice(self._prices, start, start + self._window))).T
        return self._calc_price_quotient(p)

    def _calc_price_quotient(self, p):
        for i in range(0, self._num_assets):
            lc = p[2][i][-1]
            p[0][i] /= lc
            p[1][i] /= lc
            p[2][i] /= lc

        return p

    def get_latest(self):
        n_prc = len(self._prices)
        if n_prc < self._window:
            return None, None

        p = self._make_price_tensor(n_prc - self._window)
        return p, self._portfolio

    def get_random_batch(self, size):
        num_prices = len(self._prices)
        if num_prices < self._window:
            return []

        first_possible = num_prices - self._window - size + 1
        roll = np.random.geometric(self.beta) - 1
        sel = max(first_possible - roll, 0)
        return np.array([(self._make_price_tensor(sel + i), self._portfolio)
                         for i in range(0, min(size, len(self._prices)))])

    class DataMismatchError(ValueError):
        pass