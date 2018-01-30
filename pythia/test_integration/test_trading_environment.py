import unittest

from pythia.environment.environment_wrappers import TradingEnvironment, MINIMUM_ACTION


class TradingEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.environment = TradingEnvironment(1000.0, "../test_integration/test_stock_data.csv")

    def test_first_step_holding_action(self):
        new_state, reward, done, info = self.environment.step(0.0)

        self.assertEqual([4.25, 4.31, 3.90, 4.09, 272321.0, 0, 0], new_state.tolist())
        self.assertAlmostEqual(1000.0, info)
        self.assertEqual(False, False)

    def test_first_step_selling_action(self):
        new_state, _, _, info = self.environment.step(-1.0)

        self.assertAlmostEqual(1000.0, info)
        self.assertEqual([4.25, 4.31, 3.90, 4.09, 272321.0, 0, 0], new_state.tolist())

    def test_first_step_buying_action(self):
        new_state, _, _, info = self.environment.step(1.0)

        self.assertAlmostEqual(2.76, info)
        self.assertEqual([4.25, 4.31, 3.90, 4.09, 272321.0, 233, 4.28], new_state.tolist())

    def test_keeps_buying_price(self):
        self.environment.step(1.0)
        new_state, _, _, _ = self.environment.step(0.0)

        self.assertEqual([4.17, 4.21, 4.05, 4.21, 126586, 233, 4.28], new_state.tolist())

    def test_buy_percentage(self):
        _, _, _, info = self.environment.step(0.5)

        self.assertAlmostEqual(503.52, info)

    def test_do_not_buy_at_minimum_action(self):
        _, _, _, info = self.environment.step(MINIMUM_ACTION)

        self.assertAlmostEqual(1000.0, info)

    def test_step_moves_forward_in_time(self):
        self.environment.step(0.0)

        new_state, _, _, _, = self.environment.step(0.0)

        self.assertEqual([4.17, 4.21, 4.05, 4.21, 126586.0, 0, 0], new_state.tolist())

    def test_selling_with_loss(self):
        self.environment.step(1.0)

        _, _, _, info = self.environment.step(-1.0)

        self.assertEqual(993.01, info)

    def test_selling_with_profit(self):
        self.environment.step(0.0)
        self.environment.step(0.0)
        self.environment.step(1.0)

        _, _, _, info = self.environment.step(-1.0)

        self.assertAlmostEqual(1021.51, info)

    def test_done_when_time_series_ends(self):
        self.step_right_before_the_end()

        _, _, done, _ = self.environment.step(0.0)

        self.assertEqual(True, done)

    def step_right_before_the_end(self, start_index=0):
        for _ in range(start_index, len(self.environment.stock_data.data) - 2):
            self.environment.step(0.0)

    def test_reset(self):
        self.environment.step(0.0)
        self.environment.step(1.0)
        self.environment.step(-1.0)
        self.environment.step(1.0)

        first_state = self.environment.reset()

        self.assertEqual([4.28, 4.38, 4.15, 4.25, 101970, 0, 0], first_state.tolist())
        self.assertEqual(self.environment.wealth, self.environment.portfolio)
        self.assertEqual(self.environment.previous_wealth, self.environment.portfolio)
        self.assertEqual(0, len(self.environment.actions))
        self.assertEqual(0, self.environment.buying_price)

    def test_selling_at_threshold(self):
        self.environment.step(1.0)
        _, _, _, info = self.environment.step(-MINIMUM_ACTION)

        self.assertAlmostEqual(2.76, info)

    def test_positive_reward_when_finished(self):
        self.environment.step(0.0)
        self.environment.step(0.0)
        self.environment.step(1.0)

        _, reward, _, _ = self.environment.step(-1.0)

        self.assertAlmostEqual(21.51, reward)

    def test_negative_reward_when_finished(self):
        self.environment.step(1.0)

        _, reward, _, _ = self.environment.step(-1.0)

        self.assertAlmostEqual(-6.99, reward)

    def test_buying_has_no_reward(self):
        _, reward, _, _ = self.environment.step(1.0)

        self.assertAlmostEqual(0.0, reward)
