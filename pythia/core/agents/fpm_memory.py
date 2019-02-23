import numpy as np


def _offset_index(index, offset):
    if isinstance(index, slice):
        return slice(index.start + offset, index.stop + offset, index.step)
    return index + offset


def _modulo_index(index, modulo):
    if isinstance(index, slice):
        index = slice(index.start % modulo, index.stop % modulo, index.step)
        if index.stop < index.start:
            return [i for i in range(index.start, modulo)] + [i for i in range(0, index.stop)]
        return index
    return index % modulo


class Storage:
    class Proxy:
        def __init__(self, storage, index):
            self._storage = storage
            self._state_index = self._wrap_index(index)
            self._symbs_index = self._wrap_index(self._offset_to_symbols(index))

        def _wrap_index(self, index):
            return _modulo_index(_offset_index(index, self._storage.offset), self._storage.capacity)

        def _offset_to_symbols(self, index):
            return _offset_index(index, self._storage.window - 1)

        @property
        def state(self):
            return self._storage.state_matrix[self._state_index]

        @property
        def portfolio(self):
            return self._storage.symbs_matrix[self._symbs_index, :, 1]

        @portfolio.setter
        def portfolio(self, value):
            self._storage.symbs_matrix[self._symbs_index, :, 1] = value

        @property
        def future(self):
            return self._storage.symbs_matrix[self._symbs_index, :, 0]

        def __iter__(self):
            yield self.state
            yield self.portfolio
            yield self.future

    def __init__(self, size, n_indicators, n_symbols, window_size):
        self.state_matrix = np.empty((size, n_indicators, n_symbols, window_size))
        self.symbs_matrix = np.empty((size, n_symbols, 2))
        self.window = window_size
        self.capacity = size
        self._num_assets = n_symbols
        self._size = 0
        self.offset = 0

    def append(self, prices, portfolio):
        index = self._size % self.capacity
        self._size += 1
        self.offset = self._size // self.capacity
        p = np.array(prices).T
        cur_w = min(self._size, self.window)
        for w_idx in range(0, cur_w):
            self.state_matrix[index - w_idx, :, :, w_idx] = p

        self.symbs_matrix[index, :, 0] = p[0, :]
        self.symbs_matrix[index, :, 1] = portfolio

        if self._size >= self.window:
            self._calc_window_quotient(index)

        if self._size > 1:
            self._calc_future_quotient(index)

    def _calc_window_quotient(self, index):
        last_w_idx = index - self.window + 1
        for i in range(0, self._num_assets):
            lc = self.state_matrix[last_w_idx, 0, i, -1]
            self.state_matrix[last_w_idx, 0, i, :] /= lc
            self.state_matrix[last_w_idx, 1, i, :] /= lc
            self.state_matrix[last_w_idx, 2, i, :] /= lc

    def _calc_future_quotient(self, index):
        self.symbs_matrix[index - 1, :, 0] = self.symbs_matrix[index, :, 0] / self.symbs_matrix[index - 1, :, 0]

    def __getitem__(self, key):
        return self.Proxy(self, key)

    def __len__(self):
        return max(self._size - self.window + 1, 0)

    @property
    def empty(self):
        return len(self) == 0


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

    def __init__(self, config):
        self._window = config["training"]["window"]
        self.beta = config["training"]["beta"]
        self._num_assets = len(config["trading"]["coins"])
        size = int(config["training"]["size"])
        self._storage = Storage(size, 3, self._num_assets, self._window)

    def record(self, prices, portfolio):
        self._validate_input(len(prices), portfolio)
        self._storage.append(prices, portfolio[1:])

    def _validate_input(self, m, portfolio):
        if m + 1 != len(portfolio):
            raise self.DataMismatchError("Amount of asset symbols and length of portfolio vector does not match: "
                                         "m={} len(portfolio)={}", m, len(portfolio))

    def get_latest(self):
        if not self.ready():
            return None, None

        idx = len(self._storage) - 1
        s, w, _ = self._storage[idx]
        return s, w

    def ready(self):
        return not self._storage.empty

    def get_random_batch(self, size):
        num_prices = len(self._storage)
        if num_prices < 2:
            return self.EMPTY_BATCH

        first_possible = num_prices - size - 1
        roll = np.random.geometric(self.beta) - 1
        selection = max(first_possible - roll, 0)
        return self._make_batch(selection, size)

    def _make_batch(self, from_idx, size):
        size = min(size, len(self._storage) - 1)
        state, weight, future = self._storage[from_idx:from_idx + size]
        return self.Batch(state, weight, future, from_idx, size)

    def update(self, batch):
        w_idx = 0
        for i in range(batch.index + 1, batch.index + batch.size + 1):
            self._storage[i].portfolio = batch.predictions[w_idx][1:]
            w_idx += 1

    class DataMismatchError(ValueError):
        pass
