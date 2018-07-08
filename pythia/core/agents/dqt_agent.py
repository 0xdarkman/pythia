import random
from collections.__init__ import deque

import numpy as np


class DQTAgent:
    def __init__(self, model_factory, actions, alpha, gamma, batch_size, memory, interpolation):
        self.model_target = model_factory()
        self.model = model_factory()
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.interpolation = interpolation
        self.memory = Memory(memory, batch_size)

    def start(self, state):
        q = self.model_target.predict(state)
        self.memory.record_state(state)
        return self._select_action(q)

    def step(self, next_state, rewards):
        q = self.model_target.predict(next_state)
        self.memory.record_signal(self._calc_signal(q, rewards))
        if self.memory.is_ready():
            self.model.train(self.memory.get_batch())
            self.model_target.interpolate(self.model, self.interpolation)
        self.memory.record_state(next_state)
        return self._select_action(q)

    def _calc_signal(self, qn, rewards):
        qp = self.model.predict(self.memory.get_last_state())
        signals = self.alpha * (np.array(rewards) + (self.gamma * qn) - qp)
        return signals

    def _select_action(self, q):
        last_a = self.memory.get_last_action()
        if last_a is not None:
            q[self.actions.index(last_a)] = float("-inf")

        a = self.actions[np.argmax(q)]
        self.memory.record_action(a)
        return a

    def finish(self, _):
        pass


class Memory:
    def __init__(self, size, batch_size):
        self.batch_size = batch_size
        self.states = deque(maxlen=size)
        self.signals = deque(maxlen=size)
        self.action = None

    def record_state(self, state):
        self.states.append(state)

    def record_signal(self, signal):
        self.signals.append(signal)

    def record_action(self, action):
        self.action = action

    def is_ready(self):
        return len(self) >= self.batch_size

    def get_batch(self):
        roll = random.randint(0, len(self) - self.batch_size)
        return {"inputs": np.array(self.states)[roll:roll + self.batch_size],
                "targets": np.array(self.signals)[roll:roll + self.batch_size]}

    def get_last_state(self):
        return self.states[-1]

    def get_last_action(self):
        return self.action

    def __len__(self):
        return len(self.signals)

class DQTStateTransformerBase:
    def __init__(self, exchange):
        self.exchange = exchange

    def make_rates_array(self, raw_state):
        return np.array([r[self.exchange].close for r in raw_state["rates"]])


class DQTDiffStateTransformer(DQTStateTransformerBase):
    def __init__(self, exchange):
        super().__init__(exchange)

    def __call__(self, raw_state):
        s = np.diff(self.make_rates_array(raw_state))
        return s


class DQTRatioStateTransformer(DQTStateTransformerBase):
    def __init__(self, exchange):
        super().__init__(exchange)

    def __call__(self, raw_state):
        a = self.make_rates_array(raw_state)
        return a[1:] / a[0]


class DQTRewardCalc:
    def __init__(self, n, exchange):
        self.n = n
        self.exchange = exchange

    def __call__(self, state):
        r = state["rates"]
        zt = r[-1][self.exchange].close
        ztt = r[-2][self.exchange].close
        ztn = r[-(self.n + 1)][self.exchange].close
        f = (zt - ztt) / ztt
        fn = ztt / ztn
        return [(1 - f) * fn, fn, (1 + f) * fn]