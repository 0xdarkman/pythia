import pytest

import pythia.core.streams.fpm_time_series as sut
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
        self.received_reset_call = False

    def get_next_prices(self, cash, symbol):
        self.received_price_queries.append((cash, symbol,))
        return super().get_next_prices(cash, symbol)

    def reset(self):
        self.received_reset_call = True


class RowStub:
    def __init__(self, close, high, low):
        self.close = close
        self.high = high
        self.low = low


class DataFrameStub:
    def __init__(self, time, row):
        self.time = time
        self.row = row
        self.loc = self.LocProxy(self.time, self.row)

    class LocProxy:
        def __init__(self, time, row):
            self.time = time
            self.row = row

        def __getitem__(self, time):
            if self.time == time:
                return self.row
            return RowStub(0, 0, 0)


class PandasStub:
    def __init__(self):
        self.pairs = {}

    def for_pairs(self, cash, sym):
        k = f"{cash}_{sym}"
        self.pairs[k] = self.PairProxyStub()
        return self.pairs[k]

    def read_csv(self, file, index_col):
        assert index_col == "timestamp"
        assert file[-4:] == ".csv"
        pair = self.pairs.get(file[file.rfind('/') + 1:-4], self.PairProxyStub())
        return DataFrameStub(pair.time, pair.proxy.row)

    class TimeProxyStub:
        def __init__(self):
            self.row = RowStub(0, 0, 0)

        def do_return(self, close, high, low):
            self.row = RowStub(close, high, low)

    class PairProxyStub:
        def __init__(self):
            self.time = 0
            self.proxy = PandasStub.TimeProxyStub()

        def on(self, time):
            self.time = time
            return self.proxy


@pytest.fixture
def connection():
    return ConnectionSpy()


@pytest.fixture
def config():
    return {"start": 1551434400, "cash": "SYM0", "coins": ["SYM1", "SYM2"], "training_data_dir": "/a/directory/"}


@pytest.fixture
def start(config):
    return config["start"]


@pytest.fixture
def cash(config):
    return config["cash"]


@pytest.fixture
def coins(config):
    return config["coins"]


@pytest.fixture(autouse=True)
def pandas():
    prev_pd = sut.pd
    sut.pd = PandasStub()
    yield sut.pd
    sut.pd = prev_pd


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
def test_series_queries_connections_for_prices_of_all_configured_symbols(connection, symbols, cash, config):
    config["cash"] = cash
    config["coins"] = symbols
    series = make_series(connection, config)
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


def test_reset_invokes_reset_on_connection(connection, series):
    series.reset()
    assert connection.received_reset_call


def test_reset_returns_price_data_at_start_time_from_records(series, pandas, start, cash, coins):
    pandas.for_pairs(cash, coins[0]).on(start).do_return(2, 3, 1)
    pandas.for_pairs(cash, coins[1]).on(start).do_return(12, 15, 5)
    assert series.reset() == [[2, 3, 1], [12, 15, 5]]
