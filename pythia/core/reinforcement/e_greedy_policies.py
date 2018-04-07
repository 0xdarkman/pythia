import random

import numpy as np


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


