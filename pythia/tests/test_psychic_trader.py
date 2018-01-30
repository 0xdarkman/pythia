from pythia.environment.simulators import PsychicTrader


def make_trader(portfolio, period, threshold=0.01):
    trader = PsychicTrader(portfolio, period, threshold)
    return trader


def test_unchanged_prices_no_profit():
    trader = make_trader(10000.0, [1.0, 1.0, 1.0])
    assert trader.portfolio == 10000.0


def test_when_prices_double_trader_doubles_portfolio():
    trader = make_trader(10000.0, [1.0, 2.0])
    assert trader.portfolio == 20000.0


def test_when_prices_fall_sell_before():
    trader = make_trader(10000.0, [1.0, 2.0, 1.0])
    assert trader.portfolio == 20000.0


def test_wait_for_local_base_before_buying():
    trader = make_trader(10000.0, [3.0, 2.0, 1.0, 2.0, 1.0])
    assert trader.portfolio == 20000.0


def test_buying_maximum_number_of_shares_keeps_remaining_money():
    trader = make_trader(9.0, [2.0, 4.0])
    assert trader.portfolio == 17.0


def test_wait_for_maximum_peak_before_selling():
    trader = make_trader(10000.0, [1.0, 2.0, 3.0, 1.0])
    assert trader.portfolio == 30000.0


def test_multiple_peaks_makes_optimal_decisions():
    trader = make_trader(10000.0, [2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0])
    assert trader.portfolio == 120000.0


def test_produces_correct_buy_and_sell_points():
    trader = make_trader(10000.0, [2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 2.0, 1.0])
    assert trader.actions == [0, 1, 0, 2, 1, 0, 2, 1, 2, 0]


def test_ignore_if_change_is_below_threshold():
    trader = make_trader(10000.0, [2.0, 2.9, 1.0, 2.0, 1.8, 3.0, 1.0], 0.5)
    assert trader.portfolio == 30000.0