import unittest
from collections import deque

from pythia.reinforcement.e_greedy_policies import EpsilonGreedyPolicy
from pythia.reinforcement.n_step_sarsa import NStepSarsa
from pythia.reinforcement.q_table import QTable


class EnvironmentSim(object):
    def __init__(self):
        self.records = deque()
        self.current_records = deque()

    def record(self, state, reward):
        self.records.append((state, reward))

    def reset(self):
        self.current_records = deque(self.records)
        return self.current_records.popleft()[0]

    def step(self, action):
        s, r = self.current_records.popleft()
        return s, r, len(self.current_records) == 0, None


class EnvironmentSpy(object):
    def __init__(self, episode_length):
        self.received_actions = []
        self.episode_length = episode_length

    def reset(self):
        return 0

    def step(self, action):
        self.received_actions.append(action)
        self.episode_length -= 1
        return 0, 0, self.episode_length == 0, None


class PolicyStub(object):
    def __init__(self, *selected_actions):
        self.selected_actions = deque(selected_actions)

    def select(self, state):
        return self.selected_actions.popleft()


class NStepSarsaTests(unittest.TestCase):
    def setUp(self):
        self.env = EnvironmentSim()
        self.policy_factory = self.make_epsilon_greedy_policy

    def test_td_zero_one_action_one_state(self):
        tdn = self.make_tdn([0])
        self.record_states((0, 0), (1, 2))

        tdn.run()

        self.assertAlmostEqual(2, self.q_table[0, 0])

    def make_tdn(self, action_space, gamma=1.0, alpha=1.0, steps=1):
        self.q_table = QTable(action_space)
        self.policy = self.policy_factory(self.q_table)
        tdn = NStepSarsa(self.env, self.q_table, self.policy, action_space)
        tdn.steps = steps
        tdn.gamma = gamma
        tdn.alpha = alpha
        return tdn

    def make_epsilon_greedy_policy(self, q_table):
        return EpsilonGreedyPolicy(q_table, 0.0)

    def record_states(self, *state_rewards):
        for sr in state_rewards:
            self.env.record(sr[0], sr[1])

    def test_multiple_states_no_back_propagation(self):
        tdn = self.make_tdn([0], 0)
        self.record_states((0, 0), (1, 2), (2, -1))

        tdn.run()

        self.assertAlmostEqual(2, self.q_table[0, 0])
        self.assertAlmostEqual(-1, self.q_table[1, 0])

    def test_back_propagation(self):
        tdn = self.make_tdn([0], 0.5, 0.5)
        self.record_states((0, 0), (1, 2), (2, -1))

        tdn.run()

        self.assertAlmostEqual(1, self.q_table[0, 0])
        self.assertAlmostEqual(-0.5, self.q_table[1, 0])

        tdn.run()

        self.assertAlmostEqual(1.375, self.q_table[0, 0])
        self.assertAlmostEqual(-0.75, self.q_table[1, 0])

    def test_actions_follow_policy(self):
        self.env = EnvironmentSpy(4)
        self.policy_factory = lambda q: PolicyStub(1, 1, 0, 1)
        tdn = self.make_tdn([0, 1])

        tdn.run()

        self.assertEqual(1, self.env.received_actions[0])
        self.assertEqual(1, self.env.received_actions[1])
        self.assertEqual(0, self.env.received_actions[2])
        self.assertEqual(1, self.env.received_actions[3])

    def test_correct_action_values_are_updated(self):
        self.policy_factory = lambda q: PolicyStub(0, 1)
        tdn = self.make_tdn([0, 1])
        self.record_states((0, 0), (1, 2), (2, 3))

        tdn.run()

        self.assertAlmostEqual(2, self.q_table[0, 0])
        self.assertAlmostEqual(3, self.q_table[1, 1])

    def test_steps_greater_one(self):
        tdn = self.make_tdn([0], 1.0, 1.0, 2)
        self.record_states((0, 0), (1, 2), (2, -1), (3, 0), (4, 4))

        tdn.run()

        self.assertAlmostEqual(1, self.q_table[0, 0])
        self.assertAlmostEqual(-1, self.q_table[1, 0])
        self.assertAlmostEqual(4, self.q_table[2, 0])
        self.assertAlmostEqual(4, self.q_table[3, 0])

    def test_gamma_falloff(self):
        tdn = self.make_tdn([0], 0.5, 1.0, 3)
        self.q_table[3, 0] = 2
        self.record_states((0, 0), (1, 2), (2, -1), (3, 3), (4, 4))

        tdn.run()

        self.assertAlmostEqual(2.5, self.q_table[0, 0])
        self.assertAlmostEqual(1.5, self.q_table[1, 0])
        self.assertAlmostEqual(5, self.q_table[2, 0])
        self.assertAlmostEqual(4, self.q_table[3, 0])
