import itertools
import random
from functools import reduce

from collections.__init__ import deque

from pythia.core.streams.shape_shift_rates import interim_lookahead, calculate_exchange_max_differences

STOP_AT_THRESHOLD = "stop_at_threshold"


class RiggedPolicy:
    def __init__(self, env, inner_policy, rigging_chance, threshold=None, rigging_distance=None):
        """
        The rigged policy forces lucrative coin exchanges instead of completely random action selection dependent
        on finding valid lucrative exchanges and the rigging chance rate

        :param env: Agent environment
        :param inner_policy: Policy used when random actions are not rigged
        :param rigging_chance: Chance a random action will be rigged
        :param rigging_distance: Distance to look ahead for valid lucrative exchanges
        """
        self.env = env
        self.inner_policy = inner_policy
        self.rigging_chance = rigging_chance
        self.threshold = threshold
        self.rigging_distance = rigging_distance
        self.rigged_actions = deque()
        self.rigging_count = 0

    def select(self, state, q_function):
        if len(self.rigged_actions) == 0:
            if random.random() < self.rigging_chance:
                self.rigging_count += 1
                self.rigged_actions = self._make_rigged_actions()

        if len(self.rigged_actions) != 0:
            return self.rigged_actions.popleft()
        return self.inner_policy.select(state, q_function)

    def _make_rigged_actions(self):
        with interim_lookahead(self.env.rates_stream):
            target_diff = self.threshold if self.rigging_distance == STOP_AT_THRESHOLD else None
            exchange_ranges = calculate_exchange_max_differences(
                self._take_up_to_distance(self.env.rates_stream), target_diff)
        if len(exchange_ranges) == 0:
            return deque()

        exchanges = self._get_possible_exchanges()
        best_exchange = self._find_best_exchange(exchange_ranges, exchanges)
        if best_exchange is None:
            return deque()
        if self.threshold is not None and best_exchange.max_difference < self.threshold:
            return deque()

        return deque(self._make_rigged_actions_sequence(best_exchange))

    def _take_up_to_distance(self, rates_stream):
        if self.rigging_distance is None or self.rigging_distance == STOP_AT_THRESHOLD:
            return rates_stream
        return itertools.islice(rates_stream, self.rigging_distance)

    def _get_possible_exchanges(self):
        def make_exchange_pair(target_coin):
            return self.env.coin + "_" + target_coin

        def is_not_active(coin):
            return coin != self.env.coin

        return list(map(make_exchange_pair, filter(is_not_active, self.env.action_to_coin.values())))

    @staticmethod
    def _find_best_exchange(exchanges, targets):
        def positive_exchanges(t):
            return exchanges[t].max_difference > 0

        def to_biggest_difference(max_e, t):
            e = exchanges[t]
            if max_e is None:
                return e
            if e.max_difference > max_e.max_difference:
                return e
            return max_e

        return reduce(to_biggest_difference, filter(positive_exchanges, targets), None)

    def _make_rigged_actions_sequence(self, exchange):
        exchange_to = self._coin_to_action(exchange.name.split('_')[1])
        exchange_back = self._coin_to_action(self.env.coin)
        start_pos = exchange.start_position
        diff_pos = exchange.end_position - start_pos
        return [0] * start_pos + [exchange_to] + [0] * (diff_pos - 1) + [exchange_back]

    def _coin_to_action(self, coin):
        for k, v in self.env.action_to_coin.items():
            if v == coin:
                return k