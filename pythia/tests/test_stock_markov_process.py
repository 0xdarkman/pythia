import unittest

from pythia.environment.stocks import StockData
from pythia.reinforcement.stock_markov_process import StockMarkovProcess


class TradingEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.process = StockMarkovProcess(StockData("test_model_data.csv"))

    def test_first_step_states(self):
        states = self.process.get_states_of_step(0)

        self.assertListEqual([[3, 0, 0, 0]], states)

    def test_second_step_states(self):
        states = self.process.get_states_of_step(1)

        self.assertListEqual([[5, 0, 1, 0], [5, 3, 1, 0]], states)

    def test_third_step_states(self):
        states = self.process.get_states_of_step(2)

        self.assertListEqual([[4, 0, 2, 0], [4, 0, 2, 2], [4, 3, 2, 0], [4, 5, 2, 0]], states)

    def test_fourth_step_states(self):
        states = self.process.get_states_of_step(3)

        self.assertListEqual([[1, 0, 3, 0], [1, 0, 3, 2], [1, 4, 3, 2], [1, 0, 3, 1], [1, 0, 3, -1], [1, 3, 3, 0], [1, 5, 3, 0], [1, 4, 3, 0]], states)

'''
    def test_fifth_step_states(self):
        states = self.process.get_states_of_step(4)

        self.assertListEqual([[2, 0, 4, 0], [2, 0, 4, 2], [2, 4, 4, 2], [2, 0, 4, 1], [2, 0, 4, -1], [2, 1, 4, 2], [2, 0, 4, -3], [2, 1, 4, 1], [2, 1, 4, -1], [2, 0, 4, -2], [2, 0, 4, -4],
                              [2, 3, 4, 0], [2, 5, 4, 0], [2, 4, 4, 0], [2, 1, 4, 0]], states)



    def test_second_last_step_states(self):
        states = self.process.get_states_of_step(6)

        self.assertListEqual([[6, 0, 6], [6, 3, 6], [6, 5, 6], [6, 4, 6], [6, 1, 6], [6, 2, 6]], states)

    def test_terminal_state(self):
        states = self.process.get_states_of_step(7)

        self.assertListEqual([[4, 0, 7], [4, 3, 7], [4, 5, 7], [4, 4, 7], [4, 1, 7], [4, 2, 7], [4, 6, 7]], states)

    def test_get_all_states(self):
        all_states = self.process.get_all_states()

        self.assertListEqual([[3, 0, 0],
                              [5, 0, 1], [5, 3, 1],
                              [4, 0, 2], [4, 3, 2], [4, 5, 2],
                              [1, 0, 3], [1, 3, 3], [1, 5, 3], [1, 4, 3],
                              [2, 0, 4], [2, 3, 4], [2, 5, 4], [2, 4, 4], [2, 1, 4],
                              [5, 0, 5], [5, 3, 5], [5, 5, 5], [5, 4, 5], [5, 1, 5], [5, 2, 5],
                              [6, 0, 6], [6, 3, 6], [6, 5, 6], [6, 4, 6], [6, 1, 6], [6, 2, 6],
                              [4, 0, 7], [4, 3, 7], [4, 5, 7], [4, 4, 7], [4, 1, 7], [4, 2, 7], [4, 6, 7]], all_states)

    def test_get_state_and_reward_of_hold_action(self):
        reward, next_state, done = self.process.action_at_state([5, 0, 1], 0)
        self.assertEqual(0, reward)
        self.assertEqual([4, 0, 2], next_state)
        self.assertFalse(done)

        reward, next_state, done = self.process.action_at_state([2, 5, 4], 0)
        self.assertEqual(0, reward)
        self.assertEqual([5, 5, 5], next_state)
        self.assertFalse(done)

        reward, next_state, done = self.process.action_at_state([4, 2, 7], 0)
        self.assertEqual(0, reward)
        self.assertEqual([], next_state)
        self.assertTrue(done)

    def test_get_state_and_reward_of_buy_action(self):
        reward, next_state, done = self.process.action_at_state([5, 3, 1], 1)
        self.assertEqual(0, reward)
        self.assertEqual([4, 3, 2], next_state)
        self.assertFalse(done)

        reward, next_state, done = self.process.action_at_state([5, 0, 5], 1)
        self.assertEqual(0, reward)
        self.assertEqual([6, 5, 6], next_state)
        self.assertFalse(done)

        reward, next_state, done = self.process.action_at_state([4, 0, 7], 1)
        self.assertEqual(0, reward)
        self.assertEqual([], next_state)
        self.assertTrue(done)

    def test_get_state_and_reward_of_sell_action(self):
        reward, next_state, done = self.process.action_at_state([5, 0, 1], -1)
        self.assertEqual(0, reward)
        self.assertEqual([4, 0, 2], next_state)
        self.assertFalse(done)

        reward, next_state, done = self.process.action_at_state([5, 5, 5], -1)
        self.assertEqual(1, reward)
        self.assertEqual([6, 0, 6], next_state)
        self.assertFalse(done)

        reward, next_state, done = self.process.action_at_state([4, 5, 2], -1)
        self.assertEqual(-4, reward)
        self.assertEqual([1, 0, 3], next_state)
        self.assertFalse(done)

        reward, next_state, done = self.process.action_at_state([4, 5, 7], -1)
        self.assertEqual(-1, reward)
        self.assertEqual([], next_state)
        self.assertTrue(done)
'''
