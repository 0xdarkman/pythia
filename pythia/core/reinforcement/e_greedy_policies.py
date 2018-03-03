import random
import numpy as np


class EpsilonGreedyPolicy(object):
    def __init__(self, q_function, epsilon):
        self.q_function = q_function
        self.epsilon = epsilon

    def select(self, state):
        if callable(self.epsilon):
            e = self.epsilon()
        else:
            e = self.epsilon

        action_space = self.q_function.action_space
        if random.random() < e:
            return random.choice(action_space)

        vs = self.q_function.get_action_values_of_state(state)
        return action_space[np.argmax(vs)]


class NormalEpsilonGreedyPolicy(object):
    def __init__(self, q_function, epsilon):
        self.q_function = q_function
        self.epsilon = epsilon

    def select(self, state):
        if callable(self.epsilon):
            e = self.epsilon()
        else:
            e = self.epsilon

        action_space = self.q_function.action_space
        vs = self.q_function.get_action_values_of_state(state)
        return action_space[np.argmax(vs + np.random.randn(1, len(action_space)) * e)]
