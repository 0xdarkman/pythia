import random
import numpy as np


class EpsilonGreedyPolicy(object):
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

        vs = q_function.get_action_values_of_state(state)
        return action_space[np.argmax(vs)]


class NormalEpsilonGreedyPolicy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select(self, state, q_function):
        if callable(self.epsilon):
            e = self.epsilon()
        else:
            e = self.epsilon

        action_space = q_function.action_space
        vs = q_function.get_action_values_of_state(state)
        return action_space[np.argmax(vs + np.random.randn(1, len(action_space)) * e)]
