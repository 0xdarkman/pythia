import pandas as pd
import pytest

from pythia.core.streams.fpm_time_series import FpmTimeSeries


def make_series(cash, *symbols):
    return FpmTimeSeries(cash, *symbols)


def uniform_data(*values):
    return pd.DataFrame({'close': values, 'high': values, 'low': values})


def uniform_prices(*values):
    return [[v, v, v] for v in values]


def data(*values):
    cl = []
    hl = []
    ll = []
    for c, h, l in values:
        cl.append(c)
        hl.append(h)
        ll.append(l)
    return pd.DataFrame({'close': cl, 'high': hl, 'low': ll})


def prices(*values):
    return [[c, h, l] for c, h, l in values]


def test_creation():
    assert make_series(uniform_data(1), uniform_data(1))


def test_time_series_needs_at_least_one_symbol():
    with pytest.raises(FpmTimeSeries.NoSymbolsError):
        make_series(uniform_data(1))


@pytest.mark.parametrize("cash,symbol", [(2, 1), (1, 3)])
def test_prices_are_given_in_defined_cash(cash, symbol):
    s = make_series(uniform_data(cash), uniform_data(symbol))
    assert next(s) == uniform_prices(symbol / cash)


def test_produces_correct_high_and_low_values_based_on_cash_prices():
    s = make_series(data((2, 3, 1)), data((1, 2, 3)))
    assert next(s) == prices((1 / 2, 2 / 3, 3 / 1))


def test_can_handle_multiple_assets():
    s = make_series(uniform_data(2), uniform_data(1), uniform_data(3))
    assert next(s) == uniform_prices(1 / 2, 3 / 2)


def test_can_handle_multiple_series_entries():
    s = make_series(uniform_data(2, 1), uniform_data(1, 2))
    assert next(s) == uniform_prices(1 / 2)
    assert next(s) == uniform_prices(2)


def test_raise_stop_iteration_when_reaching_end_of_series():
    s = make_series(uniform_data(1), uniform_data(1))
    next(s)
    with pytest.raises(StopIteration):
        next(s)


def test_can_reset_series():
    s = make_series(uniform_data(1), uniform_data(1))
    next(s)
    s.reset()
    next(s)
