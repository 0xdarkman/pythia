import unittest

from pythia.environment.simulators import PsychicTrader


def make_trader(portfolio, period, threshold=0.01):
    trader = PsychicTrader(portfolio, period, threshold)
    return trader


class PsychicTraderTests(unittest.TestCase):
    def test_unchanged_prices_no_profit(self):
        trader = make_trader(10000.0, [1.0, 1.0, 1.0])

        self.assertEqual(10000.0, trader.portfolio)

    def test_when_prices_double_trader_doubles_portfolio(self):
        trader = make_trader(10000.0, [1.0, 2.0])

        self.assertEqual(20000.0, trader.portfolio)

    def test_when_prices_fall_sell_before(self):
        trader = make_trader(10000.0, [1.0, 2.0, 1.0])

        self.assertEqual(20000.0, trader.portfolio)

    def test_wait_for_local_base_before_buying(self):
        trader = make_trader(10000.0, [3.0, 2.0, 1.0, 2.0, 1.0])

        self.assertEqual(20000.0, trader.portfolio)

    def test_buying_maximum_number_of_shares_keeps_remaining_money(self):
        trader = make_trader(9.0, [2.0, 4.0])

        self.assertEqual(17.0, trader.portfolio)

    def test_wait_for_maximum_peak_before_selling(self):
        trader = make_trader(10000.0, [1.0, 2.0, 3.0, 1.0])

        self.assertEqual(30000.0, trader.portfolio)

    def test_multiple_peaks_makes_optimal_decisions(self):
        trader = make_trader(10000.0, [2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0])

        self.assertEqual(120000.0, trader.portfolio)

    def test_produces_correct_buy_and_sell_points(self):
        trader = make_trader(10000.0, [2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0])

        self.assertEqual([0, 1, 0, 2, 1, 0, 2, 1, 2, 0], trader.actions)

    def test_ignore_if_change_is_below_threshold(self):
        trader = make_trader(10000.0, [2.0, 2.9, 1.0, 2.0, 1.8, 3.0, 1.0], 0.5)

        self.assertEqual(30000.0, trader.portfolio)


if __name__ == '__main__':
    unittest.main()
