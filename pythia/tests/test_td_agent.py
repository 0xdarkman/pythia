from collections import deque

import pytest


class TDAgent:
    def __init__(self, policy, q_function, n):
        self.policy = policy
        self.q = q_function
        self._record = deque()

    def start(self, state):
        a = self.policy.select(state, self.q)
        self._record.append((state, a))
        return a

    def finish(self, reward):
        s, a = self._record.popleft()
        self.q.learn(s, a, reward)


class QSpy:
    def __init__(self):
        self.learned = list()

    def learn(self, state, action, signal):
        self.learned.append(Learned(state, action, signal))


class Learned:
    def __init__(self, state, action, signal):
        self.s = state
        self.a = action
        self.r = signal

    def __eq__(self, other):
        return self.s == other.s and \
               self.a == other.a and \
               self.r == other.r

    def __repr__(self):
        return "Learned: State={}, Action={}, Signal={}".format(self.s, self.a, self.r)


class PolicyStub:
    def __init__(self):
        self.t = -1

    def select(self, state, q_function):
        self.t += 1
        return Action(self.t)


class Action:
    def __init__(self, t):
        self.t = t

    def __eq__(self, other):
        return self.t == other.t

    def __repr__(self):
        return "a{}".format(self.t)


class State:
    def __init__(self, t):
        self.t = t

    def __eq__(self, other):
        return self.t == other.t

    def __repr__(self):
        return "s{}".format(self.t)


@pytest.fixture
def q_spy():
    return QSpy()


def make_agent(n, q=None):
    q = QSpy() if q is None else q
    return TDAgent(PolicyStub(), q, n)


def test_agent_start():
    agent = make_agent(n=1)
    a = agent.start(State(0))
    assert a.t == 0


def test_immediate_finish(q_spy):
    agent = make_agent(q=q_spy, n=1)
    agent.start(State(0))
    agent.finish(10)
    assert q_spy.learned[0] == Learned(State(0), Action(0), 10)
