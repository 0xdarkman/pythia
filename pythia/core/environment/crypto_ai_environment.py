from decimal import Decimal
from collections import deque

from pythia.core.environment.crypto_environment import RatesEnvironment, EnvironmentFinished
from pythia.core.streams.shape_shift_rates import calculate_exchange_ranges


def _make_token_to_index(action_mapping, start_token):
    token_index = dict()
    i = 0
    token_index[start_token] = i
    for token in action_mapping.values():
        if token not in token_index:
            i += 1
            token_index[token] = i
    return token_index


class RatesAiEnvironment(RatesEnvironment):
    def __init__(self, rates, start_token, start_amount, window_size, action_to_token, reward_calc, exchange_filter=None):
        """
        Environment representing token rates exchanges. Represents rates, current wallet balance, and currently held
        token in an AI friendly format. Implements a mechanism to perform a token exchange by specifying the index of the
        requested token. Floating point data like rates and current balance are normalized

        :param rates: source stream containing market information for token exchanges
        :param start_token: crypto token the starting balance is held in
        :param start_amount: the starting balance
        :param window_size: the size of the moving rates window
        :param action_to_token: dictionary that maps action indices to token strings like {0:"BTC", 1:"ETH")
        :param reward_calc: callable object (RatesAiEnvironment):float that calculates a reward given the environment
        :param exchange_filter: (optional) list that filters the rates in the state showing only the coins specified
        """
        self.exchange_ranges = calculate_exchange_ranges(rates)
        self.action_to_token = action_to_token
        self.token_to_index = _make_token_to_index(self.action_to_token, start_token)
        self.reward_calc = reward_calc
        self.exchange_filter = exchange_filter
        self.starting_balance = Decimal(start_amount)
        num_exchanges = len(self.exchange_ranges) if exchange_filter is None else len(exchange_filter)
        self.window = deque([], maxlen=(window_size * num_exchanges))
        self.prev_state = None
        self.state = None
        rates.reset()
        super().__init__(rates, start_token, start_amount)

    def reset(self):
        self.window.clear()
        self._append_normalized_pairs(super().reset()[1])
        self._fill_window()
        return self._next_ai_state()

    def _next_ai_state(self):
        self.prev_state = self.state
        self.state = [self.token_to_index[self.token], self.normalized_balance] + list(self.window)
        return self.state

    @property
    def normalized_balance(self):
        b = self.balance_in(self._start_token)
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
        a = self.action_to_token[action] if action in self.action_to_token else None
        s, _, done, _ = super().step(a)
        self._append_normalized_pairs(s[1])
        return self._next_ai_state(), self.reward_calc(self), done, _


class WindowError(Exception):
    pass


class ActionFilter:
    def __init__(self, env):
        self.env = env

    def __call__(self, state_action):
        s, a = state_action
        if a not in self.env.action_to_token:
            return True

        return s[0] != self.env.token_to_index[self.env.action_to_token[a]]