import numpy as np
import pytest

from pythia.environment.stocks import StockData, ExceededStockData


@pytest.fixture
def stocks():
    return StockData("test_stock_data.csv")


def test_get_period_returns_sequence_of_stock_data(stocks):
    period = stocks.get_period(1, 3)
    assert np.array_equal(np.array([[4.25, 4.31, 3.90, 4.09, 272321],
                                    [4.17, 4.21, 4.05, 4.21, 126586],
                                    [4.26, 4.30, 4.11, 4.18, 70590]]),
                          period)


def test_period_exceeds_data_throws_exception(stocks):
    with pytest.raises(ExceededStockData):
        stocks.get_period(20, 10)
