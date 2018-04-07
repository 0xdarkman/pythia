import pytest

from pythia.core.reinforcement.td_agent import TDAgent


class QFake:
    def __init__(self):
        self.prediction = Predictions(default=0.0)
        self.learned = LearnedContainer()

    def learn(self, state, action, signal):
        self.learned.append(Learned(state, action, signal))

    def __getitem__(self, state_action):
        return self.prediction[state_action]


class Predictions(dict):
    def __init__(self, default):
        super().__init__()
        self.default = default

    def __getitem__(self, state_action):
        s, a = state_action
        return self.get((s.t, a.t), self.default)

    def __setitem__(self, state_action, value):
        s, a = state_action
        super().__setitem__((s.t, a.t), value)


class LearnedContainer(list):
    @property
    def last(self):
        return self[-1]

    @property
    def previous(self):
        return self[-2]

    @property
    def nothing(self):
        return len(self) == 0


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
        return "(Learned: State={}, Action={}, Signal={})".format(self.s, self.a, self.r)


class PolicyStub:
    def __init__(self):
        self.t = -1

    def select(self, state, q_function):
        self.t += 1
        return Action(self.t)

    def reset(self):
        self.t = -1


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
def q():
    return QFake()


@pytest.fixture
def policy():
    return PolicyStub()


@pytest.fixture
def n1_agent(q):
    return make_agent(n=1, q=q)


@pytest.fixture
def n2_agent(q):
    return make_agent(n=2, q=q)


def make_agent(n, q=None, p=None, gamma=1.0, alpha=1.0):
    q = QFake() if q is None else q
    p = PolicyStub() if p is None else p
    return TDAgent(p, q, n, gamma, alpha)


def test_agent_start(n1_agent):
    a = n1_agent.start(State(0))
    assert a.t == 0


def test_immediate_finish(n1_agent):
    n1_agent.start(State(0))
    n1_agent.finish(10)
    assert n1_agent.q.learned.last == Learned(State(0), Action(0), signal=10)


def test_start_resets_agent_state(q, policy):
    agent = make_agent(q=q, n=1, p=policy)
    agent.start(State(1))
    policy.reset()
    agent.start(State(2))
    agent.finish(0)
    assert q.learned.last == Learned(State(2), Action(0), signal=0)


def test_step_returns_next_action(n1_agent):
    n1_agent.start(State(0))
    assert n1_agent.step(State(1), 7) == Action(1)


def test_finishing_one_step_td_learns_previous_to_last_state(n1_agent):
    n1_agent.start(State(0))
    n1_agent.step(State(1), 7)
    n1_agent.finish(5)
    assert n1_agent.q.learned.last == Learned(State(1), Action(1), signal=5)


def test_discounted_prediction_is_part_of_reward_signal(q):
    agent = make_agent(n=1, q=q, gamma=0.5)
    agent.start(State(0))
    q.prediction[State(1), Action(1)] = 5.0
    agent.step(State(1), 10.0)
    assert q.learned.last == Learned(State(0), Action(0), signal=12.5)


def test_not_enough_data_for_learning(n2_agent):
    n2_agent.start(State(0))
    n2_agent.step(State(1), -3)
    assert n2_agent.q.learned.nothing


def test_reward_signal_is_weighted_difference_of_decaying_reward_sum_and_previous_prediction(q):
    """
    Q(s, a) = Q(s, a) + alpha * [G - Q(s, a)]
    """
    agent = make_agent(n=2, gamma=0.5, alpha=0.1, q=q)
    agent.start(State(0))
    agent.step(State(1), -3.0)
    q.prediction[State(2), Action(2)] = 4.0
    q.prediction[State(0), Action(0)] = 1.0
    agent.step(State(2), 10.0)
    assert q.learned.last == Learned(State(0), Action(0), signal=0.1 * ((-3.0 + 0.5 * 10.0 + 0.25 * 4.0) - 1.0))


def test_handling_remaining_steps_when_finishing(q):
    agent = make_agent(n=2, gamma=0.5, q=q)
    agent.start(State(0))
    agent.step(State(1), -3.0)
    agent.step(State(2), 10.0)
    agent.finish(2.0)
    assert q.learned.previous == Learned(State(1), Action(1), signal=10.0 + 0.5 * 2.0)
    assert q.learned.last == Learned(State(2), Action(2), signal=2.0)
