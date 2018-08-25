import itertools
from collections.__init__ import deque

import numpy as np


class FPMMemory:
    class Batch:
        def __init__(self, prices, weights, future, index, size):
            self.prices = prices
            self.weights = weights
            self.future = future
            self.predictions = None
            self._index = index
            self._size = size

        @property
        def index(self):
            return self._index

        @property
        def size(self):
            return self._size

        @property
        def empty(self):
            return self.size == 0

        def __getitem__(self, idx):
            return self.prices[idx], self.weights[idx], self.future[idx]

        def __len__(self):
            return len(self.prices)

    EMPTY_BATCH = Batch(np.empty(0), np.empty(0), np.empty(0), -1, 0)

    def __init__(self, window, size, beta):
        self._window = window
        self.beta = beta

        self._prices = deque(maxlen=size)
        self._portfolios = deque(maxlen=size)
        self._num_assets = None

    def record(self, prices, portfolio):
        self._validate_input(len(prices), portfolio)
        self._prices.append(prices)
        self._portfolios.append(np.array(portfolio))

    def _validate_input(self, m, portfolio):
        if m + 1 != len(portfolio):
            raise self.DataMismatchError("Amount of asset symbols and length of portfolio vector does not match: "
                                         "m={} len(portfolio)={}", m, len(portfolio))
        if self._num_assets is None:
            self._num_assets = m

    def get_latest(self):
        if not self.ready():
            return None, None

        n_prc = len(self._prices)
        idx = n_prc - self._window
        p = self._make_price_tensor(idx)
        return p, self._portfolios[n_prc - 1]

    def ready(self):
        return len(self._prices) >= self._window

    def _make_price_tensor(self, start):
        p = np.array(list(itertools.islice(self._prices, start, start + self._window))).T
        return self._calc_price_quotient(p)

    def _calc_price_quotient(self, p):
        for i in range(0, self._num_assets):
            lc = p[0][i][-1]
            p[0][i] /= lc
            p[1][i] /= lc
            p[2][i] /= lc

        return p

    def get_random_batch(self, size):
        num_prices = len(self._prices)
        if num_prices < self._window + 1:
            return self.EMPTY_BATCH

        first_possible = num_prices - self._window - size
        roll = np.random.geometric(self.beta) - 1
        selection = max(first_possible - roll, 0)
        return self._make_batch(selection, size)

    def _make_batch(self, from_idx, size):
        size = min(size, len(self._prices) - 1)
        prices = np.empty([size, 3, self._num_assets, self._window])
        weights = np.empty([size, self._num_assets + 1])
        futures = np.empty([size, self._num_assets])
        for i in range(0, size):
            prices[i] = self._make_price_tensor(from_idx + i)
            weights[i] = self._portfolios[from_idx + i]
            futures[i] = self._make_future_prices(from_idx + i)
        return self.Batch(prices, weights, futures, from_idx, size)

    def _make_future_prices(self, batch_idx):
        future = np.array(self._prices[batch_idx + self._window])[:, 0]
        previous = np.array(self._prices[batch_idx + self._window - 1])[:, 0]
        return future / previous

    def update(self, batch):
        w_idx = 0
        for i in range(batch.index + 1, batch.index + batch.size + 1):
            self._portfolios[i] = batch.predictions[w_idx]
            w_idx += 1

    class DataMismatchError(ValueError):
        pass
