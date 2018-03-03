from collections import deque

import numpy as np

from .q_function import QFunction


class QAnn(QFunction):
    def __init__(self, model, action_space, memory_size):
        super().__init__(action_space)
        self.model = model
        self.memory_size = memory_size
        self.memory = deque()

    def __getitem__(self, state_action):
        state = self.unpack_state_action(state_action)
        return self.model.predict(np.array([state]))[0]

    def learn(self, state, action, signal):
        self.check_action(action)
        state = self.box_state(state)
        state.append(action)
        self.memory.append(state + [signal])
        if len(self.memory) == self.memory_size:
            batch = np.array(self.memory)
            self.model.train(batch[:, 0:len(state)], batch[:, -1].reshape(self.memory_size, 1))
            self.memory.popleft()
