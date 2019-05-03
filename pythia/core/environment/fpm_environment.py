import numpy as np


def take_last_closing(last_closing, next_prices):
    return last_closing


class WiggledPrice:
    def __init__(self, pow):
        self._pow = pow

    def __call__(self, last_closing, next_prices):
        shift = np.random.power(self._pow)
        mean = last_closing * shift + next_prices[:, 0] * (1 - shift)
        spread = (next_prices[:, 1] - next_prices[:, 2]) / 4
        return np.random.normal(mean, spread)


class FpmEnvironment:
    def __init__(self, time_series, config):
        self._start_cash = config["trading"]["cash_amount"]
        self._total_assets = len(config["trading"]["coins"]) + 1
        pow = config["training"].get("price_pow")
        self._action_price_fn = take_last_closing if pow is None else WiggledPrice(pow)
        self.time_series = time_series
        self.commission = config["trading"]["commission"]
        self.assets = self._make_assets()
        self.last_action = self._make_initial_action()
        self.last_y = None
        self.next_y = None
        self.next_prices = None

    def _make_assets(self):
        assets = np.zeros(self._total_assets)
        assets[0] = self._start_cash
        return assets

    def _make_initial_action(self):
        a = np.zeros(self._total_assets)
        a[0] = 1
        return a

    def reset(self):
        self.assets = self._make_assets()
        self.last_action = self._make_initial_action()
        try:
            s = self.time_series.reset()
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
        if np.random.random() <= 0.2:
            signal = np.zeros_like(action)
        signal = action - self.last_action
        return signal

    def _calc_asset_shift(self, signal):
        prices = self._action_price_fn(self.last_y, self._add_cash_prices(np.array(self.next_prices)))
        sell = np.where(signal > 0, 0, signal)
        sell_cash = ((sell / np.where(self.last_action <= 0, 1, self.last_action)) * self.assets) * prices
        available_cash = abs(np.sum(sell_cash))
        if available_cash < 0.001:
            return np.zeros(self.assets.shape)
        buy = np.where(signal < 0, 0, signal)
        buy_cash = abs(np.sum(sell_cash)) * buy / np.sum(buy)
        return (sell_cash + buy_cash) / prices

    def _deduct_commission(self, shift):
        return np.where(shift > 0, shift * (1 - self.commission), shift)

    def _calc_y_from_prices(self, prices):
        with_cash = self._add_cash_prices(np.array(prices))
        return with_cash[:, 0]

    @staticmethod
    def _add_cash_prices(y):
        return np.insert(y, 0, np.ones(y.shape[1]), axis=0)

    def _make_next_state(self, current, reward):
        self.next_prices = next(self.time_series, None)
        return current, reward, self.next_prices is None, None

    class TimeSeriesError(AttributeError):
        pass
