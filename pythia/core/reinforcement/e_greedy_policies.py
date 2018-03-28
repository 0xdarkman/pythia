import copy
import itertools
import random
from collections import deque

import numpy as np

from pythia.core.streams.shape_shift_rates import calculate_exchange_ranges, interim_lookahead


class Policy:
    def __init__(self, action_filter):
        self.action_filter = action_filter

    def _get_actions(self, state, q_function):
        if self.action_filter is None:
            return q_function.action_space

        return self._filter_invalid_actions(state, q_function)

    def _filter_invalid_actions(self, state, q_function):
        def remove_state(state_action):
            s, a = state_action
            return a

        return list(
            map(remove_state, filter(self.action_filter, map(lambda a: (state, a), q_function.action_space))))

    @staticmethod
    def _get_q_values_of_state(state, actions, q_function):
        return [q_function[state, a] for a in actions]


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon, action_filter=None):
        super().__init__(action_filter)
        self.epsilon = epsilon

    def select(self, state, q_function):
        if callable(self.epsilon):
            e = self.epsilon()
        else:
            e = self.epsilon

        action_space = self._get_actions(state, q_function)
        if random.random() < e:
            return random.choice(action_space)

        vs = self._get_q_values_of_state(state, action_space, q_function)
        return action_space[np.argmax(vs)]


class NormalEpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon, action_filter=None):
        super().__init__(action_filter)
        self.epsilon = epsilon

    def select(self, state, q_function):
        if callable(self.epsilon):
            e = self.epsilon()
        else:
            e = self.epsilon

        action_space = self._get_actions(state, q_function)
        vs = self._get_q_values_of_state(state, action_space, q_function)
        return action_space[np.argmax(vs + np.random.randn(1, len(action_space)) * e)]


class RiggedPolicy:
    def __init__(self, env, inner_policy, rigging_chance, rigging_distance=None):
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
            exchange_ranges = calculate_exchange_ranges(self._take_up_to_distance(self.env.rates_stream))
        if len(exchange_ranges) == 0:
            return deque()

        exchanges = self._get_possible_exchanges()
        best_exchange = self._find_best_exchange(exchange_ranges, exchanges)
        if best_exchange is None:
            return deque()

        return deque(self._make_rigged_actions_sequence(best_exchange, exchange_ranges))

    def _take_up_to_distance(self, rates_stream):
        if self.rigging_distance is None:
            return rates_stream
        return itertools.islice(rates_stream, self.rigging_distance)

    def _get_possible_exchanges(self):
        def make_exchange_pair(target_coin):
            return self.env.coin + "_" + target_coin

        def is_not_active(coin):
            return coin != self.env.coin

        return list(map(make_exchange_pair, filter(is_not_active, self.env.action_to_coin.values())))

    def _find_best_exchange(self, ranges, exchanges):
        best_target = None
        biggest_diff = float("-inf")
        positive_exchanges = lambda e: ranges[e].max_position < ranges[e].min_position
        for target in filter(positive_exchanges, exchanges):
            diff = ranges[target].max - ranges[target].min
            if diff > biggest_diff:
                biggest_diff = diff
                best_target = target

        return best_target

    def _make_rigged_actions_sequence(self, best_target, ranges):
        exchange_to = self._coin_to_action(best_target.split('_')[1])
        exchange_back = self._coin_to_action(self.env.coin)
        max_pos = ranges[best_target].max_position
        diff_pos = max_pos - ranges[best_target].min_position
        return [0] * max_pos + [exchange_to] + [0] * (diff_pos - 1) + [exchange_back]

    def _coin_to_action(self, coin):
        for k, v in self.env.action_to_coin.items():
            if v == coin:
                return k
