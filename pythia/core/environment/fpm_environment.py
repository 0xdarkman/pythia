import numpy as np


class FpmEnvironment:
    def __init__(self, time_series, config):
        self.time_series = time_series
        self.commission = config["trading"]["commission"]
        total_assets = len(config["trading"]["coins"]) + 1
        self.assets = np.zeros(total_assets)
        self.assets[0] = config["trading"]["cash_amount"]
        self.last_action = np.zeros(total_assets)
        self.last_action[0] = 1
        self.last_y = None
        self.next_prices = None

    def reset(self):
        self.time_series.reset()
        try:
            s = next(self.time_series)
            self.last_y = self._calc_y_from_prices(s)
            self.next_prices = next(self.time_series)
            return s
        except StopIteration:
            raise self.TimeSeriesError("The time series provided is empty.")

    def step(self, action):
        r = self._calc_reward_from(action)
        self.last_action = action
        return self._make_next_state(self.next_prices, r)

    def _calc_reward_from(self, action):
        signal = self._calc_signal(action)
        shift = self._calc_asset_shift(signal)
        self.assets += self._deduct_commission(shift)
        y = self._calc_y_from_prices(self.next_prices)
        self.last_y = y
        return np.dot(y, self.assets)

    def _calc_signal(self, action):
        signal = action - self.last_action
        return signal

    def _calc_asset_shift(self, signal):
        sell = np.where(signal > 0.001, 0, signal)
        sell_cash = ((sell / np.where(self.last_action <= 0, 1, self.last_action)) * self.assets) * self.last_y
        available_cash = abs(np.sum(sell_cash))
        if available_cash < 0.001:
            return np.zeros(self.assets.shape)
        buy = np.where(signal < -0.001, 0, signal)
        buy_cash = abs(np.sum(sell_cash)) * buy / np.sum(buy)
        return (sell_cash + buy_cash) / self.last_y

    def _deduct_commission(self, shift):
        return np.where(shift > 0, shift * (1 - self.commission), shift)

    def _calc_y_from_prices(self, prices):
        return self._add_cash_prices(np.array(prices))[:, 0]

    @staticmethod
    def _add_cash_prices(y):
        return np.insert(y, 0, np.ones(y.shape[1]), axis=0)

    def _make_next_state(self, current, reward):
        self.next_prices = next(self.time_series, None)
        return current, reward, self.next_prices is None, None

    class TimeSeriesError(AttributeError):
        pass