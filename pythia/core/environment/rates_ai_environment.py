from decimal import Decimal
from collections import deque

from pythia.core.environment.rates_environment import RatesEnvironment, EnvironmentFinished
from pythia.core.streams.rates_calculators import calculate_exchange_ranges


def _make_token_to_index(action_mapping, start_token):
    token_index = dict()
    i = 0
    token_index[start_token] = i
    for token in action_mapping.values():
        if token not in token_index:
            i += 1
            token_index[token] = i
    return token_index


class NormalizeLinearStateTransformer:
    def __init__(self, rates, token_to_index, exchange_filter, start_token, start_amount):
        self.exchange_ranges = calculate_exchange_ranges(rates)
        self.exchange_filter = exchange_filter
        self.token_to_index = token_to_index
        self.start_amount = start_amount

    def transform(self, token, balance, window):
        new_state = [self.token_to_index[token], (balance - self.start_amount) / self.start_amount]
        for pairs in window:
            def filter_rates(t):
                if self.exchange_filter is None:
                    return True
                return t[0] in self.exchange_filter

            for n, pair in filter(filter_rates, pairs.items()):
                new_state.append(self.exchange_ranges.normalize_rate(n, pair.rate))

        return new_state


class RatesAiEnvironment(RatesEnvironment):
    def __init__(self, rates, start_token, start_amount, window_size, action_to_token, reward_calc,
                 exchange_filter=None):
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
        self.action_to_token = action_to_token
        self.token_to_index = _make_token_to_index(self.action_to_token, start_token)
        self.reward_calc = reward_calc
        self.starting_balance = Decimal(start_amount)
        self.state_transformer = NormalizeLinearStateTransformer(rates, self.token_to_index, exchange_filter, start_token, self.starting_balance)
        self.window = deque([], maxlen=(window_size))
        self.prev_state = None
        self.state = None
        rates.reset()
        super().__init__(rates, start_token, start_amount)

    def reset(self):
        self.window.clear()
        self.window.append(super().reset()["rates"])
        self._fill_window()
        return self._next_ai_state()

    def _next_ai_state(self):
        self.prev_state = self.state
        self.state = self.state_transformer.transform(self.token, self.balance_in(self.start_token), self.window)
        return self.state

    def _fill_window(self):
        while not len(self.window) == self.window.maxlen:
            try:
                s, _, _, _ = super().step(None)
                self.window.append(s["rates"])
            except EnvironmentFinished:
                raise WindowError("There is not enough data to fill the window of size {}".format(self.window.maxlen))

    def step(self, action):
        a = self.action_to_token[action] if action in self.action_to_token else None
        s, _, done, _ = super().step(a)
        self.window.append(s["rates"])
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
