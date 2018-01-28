import unittest

from pythia.environment.environment_wrappers import TradingEnvironmentTableWrapper


class TradingEnvironmentTableWrapperTests(unittest.TestCase):
    def setUp(self):
        self.env = TradingEnvironmentTableWrapper(1000.0, "test_model_data.csv", 0)

    def test_reset_state(self):
        self.env.step(0)
        s = self.env.reset()
        self.assertSequenceEqual(self.make_state(3, 0, 0), s)

    def test_step_state_hold(self):
        s, _, _, _ = self.env.step(0)
        self.assertSequenceEqual(self.make_state(5, 1, 0), s)

    def test_record_buying_price(self):
        self.env.step(1)
        self.env.step(0)
        s, _, _, _, = self.env.step(0)
        self.assertSequenceEqual(self.make_state(1, 3, 3), s)

    def test_sequential_buying(self):
        self.env.step(1)
        s, _, _, _, = self.env.step(1)
        self.assertSequenceEqual(self.make_state(4, 2, 3), s)

    def test_selling_state(self):
        self.env.step(1)
        s, _, _, _, = self.env.step(2)
        self.assertSequenceEqual(self.make_state(4, 2, 0), s)

    def test_selling_state_without_shares(self):
        s, _, _, _ = self.env.step(2)
        self.assertSequenceEqual(self.make_state(5, 1, 0), s)

    def test_buying_and_holding_rewards(self):
        _, r, _, _ = self.env.step(0)
        self.assertAlmostEqual(0.0, r)
        _, r, _, _ = self.env.step(1)
        self.assertAlmostEqual(0.0, r)

    def test_selling_rewards(self):
        self.env.step(1)
        _, r, _, _ = self.env.step(2)
        self.assertAlmostEqual(2.0, r)

    def test_reward_when_selling_without_shares(self):
        _, r, _, _ = self.env.step(2)
        self.assertAlmostEqual(0.0, r)
        _, r, _, _ = self.env.step(2)
        self.assertAlmostEqual(0.0, r)

    def test_trading_episode(self):
        s = self.env.reset()
        self.assertSequenceEqual(self.make_state(3, 0, 0), s)
        s, r, _, _ = self.env.step(0)
        self.assertSequenceEqual(self.make_state(5, 1, 0), s)
        self.assertAlmostEqual(0.0, r)
        s, r, _, _ = self.env.step(1)
        self.assertSequenceEqual(self.make_state(4, 2, 5), s)
        self.assertAlmostEqual(0.0, r)
        s, r, _, _ = self.env.step(2)
        self.assertSequenceEqual(self.make_state(1, 3, 0), s)
        self.assertAlmostEqual(-1.0, r)
        s, r, _, _ = self.env.step(1)
        self.assertSequenceEqual(self.make_state(2, 4, 1), s)
        self.assertAlmostEqual(0.0, r)
        s, r, _, _ = self.env.step(0)
        self.assertSequenceEqual(self.make_state(5, 5, 1), s)
        self.assertAlmostEqual(0.0, r)
        s, r, _, _ = self.env.step(1)
        self.assertSequenceEqual(self.make_state(6, 6, 1), s)
        self.assertAlmostEqual(0.0, r)
        s, r, d, _ = self.env.step(2)
        self.assertSequenceEqual(self.make_state(4, 7, 0), s)
        self.assertAlmostEqual(5.0, r)
        self.assertTrue(d)

    def test_penalize_invalid_actions(self):
        e = TradingEnvironmentTableWrapper(1000.0, "test_model_data.csv", 10)
        _, r, _, _ = e.step(1)
        self.assertAlmostEqual(0.0, r)
        _, r, _, _ = e.step(1)
        self.assertAlmostEqual(-10.0, r)
        _, r, _, _ = e.step(2)
        self.assertAlmostEqual(1.0, r)
        _, r, _, _ = e.step(2)
        self.assertAlmostEqual(-10.0, r)

    @staticmethod
    def make_state(price, step, buying_price):
        return [float(price), float(buying_price), float(step)]


if __name__ == '__main__':
    unittest.main()
