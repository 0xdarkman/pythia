import unittest

import numpy as np

from pythia.environment.stocks import StockData, ExceededStockData


def make_stock_data():
    return StockData("../test_integration/test_stock_data.csv")


class StockDataTests(unittest.TestCase):
    def test_get_period_returns_sequence_of_stock_data(self):
        stocks = make_stock_data()

        period = stocks.get_period(1, 3)

        self.assertTrue(
            np.array_equal(np.array([[4.25, 4.31, 3.90, 4.09, 272321],
                                     [4.17, 4.21, 4.05, 4.21, 126586],
                                     [4.26, 4.30, 4.11, 4.18, 70590]]),
                           period))

    def test_period_exceeds_data_throws_exception(self):
        stocks = make_stock_data()

        with self.assertRaises(ExceededStockData):
            stocks.get_period(20, 10)


if __name__ == '__main__':
    unittest.main()
