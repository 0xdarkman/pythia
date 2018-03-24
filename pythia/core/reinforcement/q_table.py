import numpy as np

from .q_function import QFunction
from .value_table import ValueTable


class QTable(QFunction):
    def __init__(self, action_space, initializer=None):
        super().__init__(action_space)
        self.storage = ValueTable(initializer)

    def __getitem__(self, state_action):
        state = self.unpack_state_action(state_action)
        return self.storage[state]

    def __setitem__(self, state_action, value):
        state = self.unpack_state_action(state_action)
        self.storage[state] = value

    def max_value_of(self, state):
        return np.max(self.all_values_of_state(state))

    def learn(self, state, action, signal):
        self[state, action] += signal
