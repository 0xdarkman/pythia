import random
import numpy as np


class Policy:
    @staticmethod
    def _get_q_values_of_state(state, q_function):
        return [q_function[state, a] for a in q_function.action_space]


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select(self, state, q_function):
        if callable(self.epsilon):
            e = self.epsilon()
        else:
            e = self.epsilon

        action_space = q_function.action_space
        if random.random() < e:
            return random.choice(action_space)

        vs = self._get_q_values_of_state(state, q_function)
        return action_space[np.argmax(vs)]


class NormalEpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select(self, state, q_function):
        if callable(self.epsilon):
            e = self.epsilon()
        else:
            e = self.epsilon

        action_space = q_function.action_space
        vs = self._get_q_values_of_state(state, q_function)
        return action_space[np.argmax(vs + np.random.randn(1, len(action_space)) * e)]
