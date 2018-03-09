from collections import deque
from functools import reduce


class Records:
    def __init__(self, size):
        self.size = size
        self._state_actions = deque()
        self._rewards = deque()

    def record_state_action(self, s, a):
        self._state_actions.append((s, a))

    def record_reward(self, r):
        self._rewards.append(r)

    @property
    def rewards(self):
        return self._rewards

    @property
    def is_full(self):
        return len(self._state_actions) == self.size

    @property
    def is_empty(self):
        return len(self._state_actions) == 0

    def __iter__(self):
        return self

    def __next__(self):
        self._rewards.popleft()
        return self._state_actions.popleft()

    def clear(self):
        self._state_actions.clear()
        self._rewards.clear()


class TDAgent:
    def __init__(self, policy, q_function, n, gamma):
        self.policy = policy
        self.q = q_function
        self.n = n
        self.gamma = gamma

        self._record = Records(self.n)

    def start(self, state):
        self._record.clear()
        return self._select_action_and_store(state)

    def _select_action_and_store(self, s):
        a = self.policy.select(s, self.q)
        self._record.record_state_action(s, a)
        return a

    def step(self, state, reward):
        self._record.record_reward(reward)
        if self._record.is_full:
            self._learn()

        return self._select_action_and_store(state)

    def _learn(self):
        signal = self._calc_signal()
        s, a = next(self._record)
        signal += pow(self.gamma, self.n) * self.q[s, a]
        self.q.learn(s, a, signal)

    def _calc_signal(self):
        def decaying_sum(acc, enum):
            i, r = enum
            return acc + pow(self.gamma, max(i, 0)) * r

        return reduce(decaying_sum, enumerate(self._record.rewards), 0)

    def finish(self, reward):
        self._record.record_reward(reward)
        while not self._record.is_empty:
            sig = self._calc_signal()
            s, a = next(self._record)
            self.q.learn(s, a, sig)