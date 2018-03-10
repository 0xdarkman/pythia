from decimal import Decimal
from collections import deque
from functools import reduce

from pythia.core.environment.crypto_environment import CryptoEnvironment, EnvironmentFinished


class ExchangeRanges:
    def __init__(self):
        self.ranges = dict()

    def extend(self, pair):
        if pair.name not in self.ranges:
            self.ranges[pair.name] = pair
        else:
            self.ranges[pair.name].extend(pair)

        return self

    def normalize_rate(self, name, rate):
        r = self.ranges[name]
        return (float(rate) - r.min) / (r.max - r.min)

    def __len__(self):
        return len(self.ranges)


class PairRanges:
    def __init__(self, named_pair):
        name, pair = named_pair
        self.name = name
        self.min = float(pair.rate)
        self.max = float(pair.rate)

    def extend(self, other):
        assert self.name == other.name
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        return self


def _pairs_to_extremes(pairs):
    return map(lambda pair: PairRanges(pair), pairs.items())


def _calculate_exchange_ranges(rates):
    def extend_exchange_ranges(ranges, pairs):
        return reduce(lambda r, p: r.extend(p), _pairs_to_extremes(pairs), ranges)

    return reduce(extend_exchange_ranges, rates, ExchangeRanges())


class CryptoAiEnvironment(CryptoEnvironment):
    def __init__(self, rates, start_coin, start_amount, window_size, index_to_coin, reward_calc, exchange_filter=None):
        self.exchange_ranges = _calculate_exchange_ranges(rates)
        self.index_to_coin = index_to_coin
        self.coin_to_index = {v: k for k, v in self.index_to_coin.items()}
        self.reward_calc = reward_calc
        self.exchange_filter = exchange_filter
        self.starting_balance = Decimal(start_amount)
        num_exchanges = len(self.exchange_ranges) if exchange_filter is None else len(exchange_filter)
        self.window = deque([], maxlen=(window_size * num_exchanges))
        self.prev_state = None
        self.state = None
        rates.reset()
        super().__init__(rates, start_coin, start_amount)

    def reset(self):
        self.window.clear()
        self._append_normalized_pairs(super().reset()[1])
        self._fill_window()
        return self._next_ai_state()

    def _next_ai_state(self):
        self.prev_state = self.state
        self.state = [self.coin_to_index[self.coin], self.normalized_balance] + list(self.window)
        return self.state

    @property
    def normalized_balance(self):
        b = self.balance_in(self.index_to_coin[0])
        return float((b - self.starting_balance) / self.starting_balance)

    def _fill_window(self):
        while not len(self.window) == self.window.maxlen:
            try:
                s, _, _, _ = super().step(None)
                self._append_normalized_pairs(s[1])
            except EnvironmentFinished:
                raise WindowError("There is not enough data to fill the window of size {}".format(self.window.maxlen))

    def _append_normalized_pairs(self, pairs):
        def filter_rates(t):
            if self.exchange_filter is None:
                return True
            return t[0] in self.exchange_filter

        for n, pair in filter(filter_rates, pairs.items()):
            self.window.append(self.exchange_ranges.normalize_rate(n, pair.rate))

    def step(self, action):
        s, _, done, _ = super().step(self.index_to_coin[action] if action is not None else None)
        self._append_normalized_pairs(s[1])
        return self._next_ai_state(), self.reward_calc(self), done, _


class WindowError(Exception):
    pass