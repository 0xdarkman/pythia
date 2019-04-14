import pytest

from pythia.core.streams.fpm_time_series import FpmLiveSeries


def uniform_return(value):
    return {"close": value, "high": value, "low": value}


def uniform_prices(value):
    return [value, value, value]


class ConnectionStub:
    def __init__(self):
        self.should_timeout = False
        self.return_values = dict()

    def set_to_timeout(self):
        self.should_timeout = True

    def set_return_price_of(self, symbol, return_value):
        self.return_values[symbol] = return_value

    def get_next_prices(self, cash, symbol):
        if self.should_timeout:
            raise TimeoutError
        return self.return_values.get(symbol, uniform_return(0))


class ConnectionSpy(ConnectionStub):
    def __init__(self):
        super().__init__()
        self.received_price_queries = list()

    def get_next_prices(self, cash, symbol):
        self.received_price_queries.append((cash, symbol,))
        return super().get_next_prices(cash, symbol)


@pytest.fixture
def connection():
    return ConnectionSpy()


@pytest.fixture
def config():
    return {"cash": "SYM0", "coins": ["SYM1", "SYM2"]}


def make_series(connection, config):
    return FpmLiveSeries(connection, config)


@pytest.fixture
def series(connection, config):
    return make_series(connection, config)


def test_a_time_series_needs_at_least_one_symbol(connection, config):
    config["coins"] = []
    with pytest.raises(FpmLiveSeries.NoSymbolsError):
        make_series(connection, config)


def test_raises_an_error_when_connection_times_out(connection, series):
    connection.set_to_timeout()
    with pytest.raises(FpmLiveSeries.TimeoutError):
        next(series)


@pytest.mark.parametrize("cash,symbols", [("CASH1", ["SYM1"]), ("CASH2", ["SYM1", "SYM2"])])
def test_series_queries_connections_for_prices_of_all_configured_symbols(connection, symbols, cash):
    series = make_series(connection, {"cash": cash, "coins": symbols})
    next(series)
    assert connection.received_price_queries == [(cash, s) for s in symbols]


def test_series_returns_price_data_as_list_in_order_of_closing_high_low(connection, config):
    config["coins"] = ["SYM1"]
    connection.set_return_price_of("SYM1", {"close": 2, "high": 3, "low": 1})
    series = make_series(connection, config)
    assert next(series) == [[2, 3, 1]]


def test_series_returns_prices_for_all_symbols(connection, series):
    connection.set_return_price_of("SYM1", uniform_return(1))
    connection.set_return_price_of("SYM2", uniform_return(2))
    assert next(series) == [uniform_prices(1), uniform_prices(2)]
