from collections import deque

import pytest

import pythia.core.remote.poloniex_connection as sut
from pythia.core.remote.poloniex_connection import PoloniexConnection

CHART_KEYS = ("date", "high", "low", "open", "close", "volume", "quoteVolume", "weightedAverage")


def uniform_chart(value):
    return {k: value for k in CHART_KEYS}


EMPTY_CHART = [uniform_chart(0)]
FILLED_CHART = [uniform_chart(1)]


class PoloniexApiStub:
    def __init__(self):
        self.should_timeout = False
        self.chart_return_data = deque()

    def set_to_timeout(self):
        self.should_timeout = True

    def set_chart_returns(self, *data):
        self.chart_return_data = deque(data)


class PoloniexApiSpy(PoloniexApiStub):
    class ReturnChartData:
        def __init__(self, api):
            self.api = api
            self.received_calls = list()

        def __call__(self, *args, **kwargs):
            self.received_calls.append(args)
            if self.api.should_timeout:
                raise TimeoutError

            return FILLED_CHART if len(self.api.chart_return_data) == 0 else self.api.chart_return_data.popleft()

    def __init__(self):
        super().__init__()
        self.return_chart_data = self.ReturnChartData(self)


class TimeStub:
    def __init__(self, t):
        self.t = t

    def set(self, t):
        self.t = t

    def time(self, *args, **kwargs):
        return self.t


class TimeSpy(TimeStub):
    def __init__(self, t):
        super().__init__(t)
        self.sleeps = 0
        self.recorded_sleeps = list()

    def sleep(self, seconds):
        self.sleeps = seconds
        self.recorded_sleeps.append(seconds)


class RandomStub:
    def __init__(self):
        self.last_random_value = 42

    def randint(self, start, end):
        return self.last_random_value


class TelemetryStub:
    def __init__(self):
        self.last_chart_ts = dict()

    def set_last_chart_ts_of(self, pair, t):
        self.last_chart_ts[pair] = t

    def find_last_chart_ts(self, pair, default):
        return self.last_chart_ts.get(pair, default)


class TelemetrySpy(TelemetryStub):
    def __init__(self):
        super().__init__()
        self.received_chart_data = None

    def write_chart(self, chart):
        self.received_chart_data = chart


def prices_from(chart):
    return {k: chart[k] for k in ("close", "high", "low",)}


@pytest.fixture(autouse=True)
def api():
    prev_chart = sut.return_chart_data
    api = PoloniexApiSpy()
    sut.return_chart_data = lambda *args, **kwargs: api.return_chart_data(*args, **kwargs)
    yield api
    sut.return_chart_data = prev_chart


@pytest.fixture
def config():
    return {"period": 1800, "start": 1551434400, "retry": 100}


@pytest.fixture
def start(config):
    return config["start"]


@pytest.fixture
def period(config):
    return config["period"]


@pytest.fixture
def retries(config):
    return config["retry"]


@pytest.fixture(autouse=True)
def time(start, period):
    prev_time = sut.time
    sut.time = TimeSpy(start + period)
    yield sut.time
    sut.time = prev_time


@pytest.fixture(autouse=True)
def random():
    prev_rnd = sut.random
    sut.random = RandomStub()
    yield sut.random
    sut.random = prev_rnd


@pytest.fixture
def telemetry():
    return TelemetrySpy()


@pytest.fixture
def connection(telemetry, config):
    return PoloniexConnection(telemetry, config)


def assert_random_delayed_repeat(t, a, r, n):
    assert t.recorded_sleeps == [r.last_random_value] * n
    assert len(a.return_chart_data.received_calls) == n


@pytest.mark.parametrize("cash,symbol", [("CSH1", "SYM1"), ("CSH2", "SYM2")])
def test_queries_the_poloniex_api_with_starting_parameters(connection, config, api, cash, symbol, start, period):
    connection.get_next_prices(cash, symbol)
    assert api.return_chart_data.received_calls[0] == (cash, symbol, period, start + period)


def test_queries_are_repeated_when_timing_out_until_retry_count_is_reached(connection, config, api):
    api.set_to_timeout()
    with pytest.raises(PoloniexConnection.TimeoutError):
        connection.get_next_prices("CASH", "SYMBOL")

    assert len(api.return_chart_data.received_calls) == config["retry"]


def test_returns_closing_high_and_low_price_of_single_chart_return(connection, config, api, start, period):
    api.set_chart_returns([{"date": start + period, "high": 3, "low": 1, "open": 1, "close": 2, "volume": 5,
                            "quoteVolume": 10, "weightedAverage": 0.1}])
    assert connection.get_next_prices("CASH", "SYMBOL") == {"close": 2, "high": 3, "low": 1}


def test_sleeps_to_next_period_when_necessary(connection, api, time, random, start, period):
    time_to_next = int(period / 2)
    time.set(start + (period - time_to_next))
    connection.get_next_prices("CASH", "SYMBOL")
    assert time.sleeps == time_to_next + random.last_random_value


def test_add_random_delay_when_requesting_at_exact_time_of_next_interval(connection, api, time, random, start, period):
    next_interval = start + period
    time.set(next_interval)
    connection.get_next_prices("CASH", "SYMBOL")
    assert time.sleeps == random.last_random_value


def test_queries_are_repeated_with_random_delay_when_chart_data_is_not_ready_yet(connection, api, time, random):
    api.set_chart_returns(EMPTY_CHART, FILLED_CHART)
    assert connection.get_next_prices("CASH", "SYMBOL") == prices_from(FILLED_CHART[0])
    assert_random_delayed_repeat(time, api, random, 2)


def test_raise_no_data_error_when_no_chart_data_is_returned_after_n_retried(connection, api, retries):
    api.set_chart_returns(*([EMPTY_CHART] * retries))
    with pytest.raises(PoloniexConnection.NoDataError):
        connection.get_next_prices("CASH", "SYMBOL")
    assert len(api.return_chart_data.received_calls) == retries


def test_writes_received_chart_data_to_telemetry(connection, api, telemetry, start, period):
    api.set_chart_returns([{"date": start + period, "high": 3, "low": 1, "open": 1, "close": 2, "volume": 5,
                            "quoteVolume": 10, "weightedAverage": 0.1}])
    connection.get_next_prices("CASH", "SYMBOL")
    assert telemetry.received_chart_data == {"CASH_SYMBOL": {"date": start + period, "close": 2, "high": 3, "low": 1}}


def test_gets_next_interval_from_last_telemetry_chart(connection, api, telemetry, time, start, period):
    telemetry.set_last_chart_ts_of("CASH_SYMBOL", start + period)
    time.set(start + period * 2)
    connection.get_next_prices("CASH", "SYMBOL")
    assert api.return_chart_data.received_calls[0] == ("CASH", "SYMBOL", period, start + period * 2)


def test_returns_oldest_chart_when_multiple_charts_are_returned(connection, api):
    api.set_chart_returns([uniform_chart(1), uniform_chart(2)])
    assert connection.get_next_prices("CASH", "SYMBOLS") == prices_from(uniform_chart(1))


def test_returns_charts_in_order_when_multiple_charts_are_returned_from_request(connection, api):
    api.set_chart_returns([uniform_chart(1), uniform_chart(2)])
    assert connection.get_next_prices("CASH", "SYMBOLS") == prices_from(uniform_chart(1))
    assert connection.get_next_prices("CASH", "SYMBOLS") == prices_from(uniform_chart(2))


def test_writes_the_chart_that_is_returned_to_telemetry(connection, api, telemetry):
    api.set_chart_returns([uniform_chart(1), uniform_chart(2)])
    connection.get_next_prices("CASH", "SYMBOL")
    assert telemetry.received_chart_data == {"CASH_SYMBOL": {"date": 1, "close": 1, "high": 1, "low": 1}}
    connection.get_next_prices("CASH", "SYMBOL")
    assert telemetry.received_chart_data == {"CASH_SYMBOL": {"date": 2, "close": 2, "high": 2, "low": 2}}


def test_multiple_charts_are_buffered(connection, api):
    api.set_chart_returns([uniform_chart(1), uniform_chart(2)])
    connection.get_next_prices("CASH", "SYMBOL")
    assert len(api.return_chart_data.received_calls) == 1
