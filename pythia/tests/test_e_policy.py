import random
import unittest

from pythia.reinforcement.e_greedy_policies import EpsilonGreedyPolicy
from pythia.reinforcement.q_table import QTable


class EpsilonGreedyPolicyTest(unittest.TestCase):
    def setUp(self):
        self.q_function = QTable([0, 1])

    def test_epsilon_zero(self):
        policy = self.make_policy(0)

        self.q_function[[0], 0] = -1
        self.q_function[[0], 1] = 1

        self.assertEqual(1, policy.select([0]))

    def make_policy(self, epsilon):
        return EpsilonGreedyPolicy(self.q_function, epsilon)

    def test_multiple_states(self):
        policy = self.make_policy(0)

        self.q_function[[0], 0] = -1
        self.q_function[[0], 1] = 1

        self.q_function[[1], 0] = 10
        self.q_function[[1], 1] = -5

        self.assertEqual(1, policy.select([0]))

    def test_non_zero_epsilon(self):
        policy = self.make_policy(0.2)
        random.seed(1)

        self.q_function[[0], 0] = -1
        self.q_function[[0], 1] = 1

        self.assertEqual(0, policy.select([0]))

    def test_epsilon_as_function(self):
        policy = self.make_policy(lambda: 0.2)
        random.seed(1)

        self.q_function[[0], 0] = -1
        self.q_function[[0], 1] = 1

        self.assertEqual(0, policy.select([0]))

    def test_incomplete_state(self):
        policy = self.make_policy(0)
        self.q_function[[0], 0] = -1
        self.assertEqual(1, policy.select([0]))
