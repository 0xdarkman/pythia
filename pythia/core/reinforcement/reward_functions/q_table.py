import numpy as np

from .q_function import QFunction
from .value_table import ValueTable


class QTable(QFunction):
    def __init__(self, action_space, initializer=None):
        super().__init__(action_space)
        self.storage = ValueTable(initializer)

    def __getitem__(self, state_action):
        state = self._make_input_state(state_action)
        return self.storage[state]

    def __setitem__(self, state_action, value):
        state = self._make_input_state(state_action)
        self.storage[state] = value

    def learn(self, state, action, signal):
        self[state, action] += signal
