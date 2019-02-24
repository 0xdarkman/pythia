import numpy as np


class AssetViewModeller:
    def __init__(self, prices, assets, use_zero_cash=False, relative_weights=False):
        self._size = 0
        if len(prices) == 0:
            return

        self._cursor = -1
        if len(prices[0]) != len(assets):
            raise self.MismatchError(f"# Prices ({len(prices[0])}) and Assets ({len(assets)}) are not the same")

        if len(prices) != (len(assets[0]) - 1):
            raise self.MismatchError(f"There are {len(prices)} symbols for prices but only "
                                     f"{len(assets[0])} for assets - cash is needed in assets")

        self._prices = np.array(prices)
        if not relative_weights:
            cash_func = np.zeros if use_zero_cash else np.ones
            self._prices = np.concatenate((cash_func((1, self._prices.shape[1])), self._prices), axis=0)

        self._size = len(self._prices)
        self._assets = np.array(assets, dtype='float')
        sums = np.apply_along_axis(np.sum, 1, self._assets)
        for i in range(0, len(sums)):
            self._assets[i] = np.divide(self._assets[i], sums[i])
        self._assets = self._assets.T

        if relative_weights:
            self._assets = self._assets[1:]
            for i in range(0, len(self._assets)):
                self._assets[i] /= np.sum(self._assets[i])

    def __iter__(self):
        return self

    def __next__(self):
        self._cursor += 1
        if self._cursor == len(self):
            raise StopIteration
        return self._prices[self._cursor], self._assets[self._cursor]

    def __len__(self):
        return self._size

    class MismatchError(AssertionError):
        pass
