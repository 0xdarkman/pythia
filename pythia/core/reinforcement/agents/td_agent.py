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
    def __init__(self, policy, q_function, n, gamma, alpha):
        """
        Temporal difference learning agent, that implements an n-step sarsa algorithm

        :param policy: action selection policy implementation to be used
        :param q_function: the Q value function used to learn and predict the Q value
        :param n: number of steps the TD algorithm takes
        :param gamma: discounting factor of future rewards
        :param alpha: step size of signal
        """
        self.policy = policy
        self.q = q_function
        self.n = n
        self.gamma = gamma
        self.alpha = alpha

        self._record = Records(self.n)

    def start(self, state):
        self._record.clear()
        a = self.policy.select(state, self.q)
        return self._take_and_record(state, a)

    def _take_and_record(self, s, a):
        self._record.record_state_action(s, a)
        return a

    def step(self, state, reward):
        self._record.record_reward(reward)
        action = self.policy.select(state, self.q)
        if self._record.is_full:
            self._learn(state, action)

        return self._take_and_record(state, action)

    def _learn(self, s_next, a_next):
        g = self._calc_recorded_rewards_sum()
        g += pow(self.gamma, self.n) * self.q[s_next, a_next]
        s, a = next(self._record)
        self.q.learn(s, a, self._calc_signal(s, a, g))

    def _calc_recorded_rewards_sum(self):
        def decaying_sum(acc, enum):
            i, r = enum
            return acc + pow(self.gamma, max(i, 0)) * r

        return reduce(decaying_sum, enumerate(self._record.rewards), 0)

    def _calc_signal(self, s, a, signal):
        return self.alpha * (signal - self.q[s, a])

    def finish(self, reward):
        self._record.record_reward(reward)
        while not self._record.is_empty:
            g = self._calc_recorded_rewards_sum()
            s, a = next(self._record)
            self.q.learn(s, a, self._calc_signal(s, a, g))