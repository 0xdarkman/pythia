import pytest

import pythia.core.remote.poloniex_api as sut
from pythia.core.remote.poloniex_api import return_chart_data, InvalidParameterError
from pythia.tests.fpm_doubles import TimeSpy, RandomStub

DATE_2019_03_01 = 1551434416
DATE_2019_04_20 = 1555747496
A_UNIX_DATE = 1551609000
A_LATER_DATE = 1551610000


class ResultStub:
    def __init__(self, **kwargs):
        self.status = kwargs.get("status", 200)
        self.json_value = kwargs.get("json", None)

    def raise_for_status(self):
        import requests
        if self.status != requests.codes.ok:
            raise requests.exceptions.HTTPError("A Http Error")

    def json(self):
        return self.json_value


class RequestsStub:
    def __init__(self):
        self.returns_json = None
        self.status = 200

    def set_status(self, value):
        self.status = value

    def get(self, url, params=None, **kwargs):
        return ResultStub(status=self.status, json=self.returns_json)


class RequestsSpy(RequestsStub):
    def __init__(self):
        super().__init__()
        self.received_get_url = None
        self.received_get_payload = None
        self.received_get_time_out = None
        self.num_get_calls = 0

    def get(self, url, params=None, **kwargs):
        self.num_get_calls += 1
        self.received_get_url = url
        self.received_get_payload = params
        self.received_get_time_out = kwargs.get("timeout", None)
        return super().get(url, params, **kwargs)


class Repeater:
    def __init__(self, time):
        self.time = time
        self.function = None

    def repeat_in(self, range, interval):
        for i in range:
            self.time.set(i * interval)
            self.function()


@pytest.fixture(autouse=True)
def reset_globals():
    sut._previous_request_times = []


@pytest.fixture(autouse=True)
def requests():
    prev_requests = sut.requests
    sut.requests = RequestsSpy()
    yield sut.requests
    sut.requests = prev_requests


@pytest.fixture(autouse=True)
def time():
    prev_time = sut.time
    sut.time = TimeSpy(0)
    yield sut.time
    sut.time = prev_time


@pytest.fixture(autouse=True)
def random():
    prev_rnd = sut.random
    sut.random = RandomStub()
    yield sut.random
    sut.random = prev_rnd


@pytest.fixture
def repeater(time):
    return Repeater(time)


def milliseconds(v):
    return v


def to_seconds(milliseconds):
    return milliseconds * 0.001


def _make_n_calls(time, interval, range, f, *args, **kwargs):
    for i in range:
        time.set(i * interval)
        f(*args, **kwargs)


def test_return_chart_data_creates_well_formatted_http_request(requests):
    return_chart_data("CASH", "SYMBOL", 1800, DATE_2019_03_01, DATE_2019_04_20)
    assert requests.received_get_url == "https://poloniex.com/public"
    assert requests.received_get_payload == {"command": "returnChartData", "currencyPair": "CASH_SYMBOL",
                                             "period": 1800, "start": DATE_2019_03_01, "end": DATE_2019_04_20}
    assert requests.received_get_time_out == 3.05


def test_return_chart_data_returns_a_json_object_when_successful(requests):
    json = [{"Data": A_UNIX_DATE, "high": 3, "low": 1, "open": 2, "close": 2, "volume": 10, "quoteVolume": 15,
             "weightedAverage": 2}]
    requests.returns_json = json
    assert return_chart_data("CASH", "SYMBOL", 1800, DATE_2019_03_01, DATE_2019_04_20) == json


def test_return_chart_data_raises_errors_when_http_status_is_not_ok(requests):
    requests.set_status(404)
    import requests.exceptions
    with pytest.raises(requests.exceptions.RequestException):
        return_chart_data("CASH", "SYMBOL", 1800, DATE_2019_03_01, DATE_2019_04_20)


def test_return_chart_data_raises_error_when_invalid_period_is_specified(requests):
    with pytest.raises(InvalidParameterError):
        return_chart_data("CASH", "SYMBOL", 1801, DATE_2019_03_01, DATE_2019_04_20)


@pytest.mark.parametrize("period", [300, 900, 1800, 7200, 14400, 86400])
def test_return_chart_data_works_with_all_valid_periods(requests, period):
    return_chart_data("SYM0", "SYM1", period, A_UNIX_DATE, A_LATER_DATE)
    assert requests.received_get_payload == {"command": "returnChartData", "currencyPair": "SYM0_SYM1",
                                             "period": period, "start": A_UNIX_DATE, "end": A_LATER_DATE}


def test_return_chart_data_can_be_invoked_without_an_end_date(requests):
    return_chart_data("SYM0", "SYM1", 300, A_UNIX_DATE)
    assert requests.received_get_payload == {"command": "returnChartData", "currencyPair": "SYM0_SYM1",
                                             "period": 300, "start": A_UNIX_DATE}


def test_subsequent_calls_to_chart_data_are_randomly_delayed_minus_the_time_already_expired(requests, random, time):
    random.last_random_value = milliseconds(42)
    time.set(0)
    return_chart_data("CASH", "SYMBOL", 300, A_UNIX_DATE)
    time.set(to_seconds(10))
    return_chart_data("CASH", "SYMBOL", 300, A_UNIX_DATE)
    assert time.recorded_sleeps == [to_seconds(32)]


def test_subsequent_calls_are_not_delayed_additionally_when_time_between_calls_exceeds_randomly_selected_delay(requests,
                                                                                                               random,
                                                                                                               time):
    random.last_random_value = milliseconds(42)
    time.set(0)
    return_chart_data("CASH", "SYMBOL", 300, A_UNIX_DATE)
    time.set(to_seconds(42))
    return_chart_data("CASH", "SYMBOL", 300, A_UNIX_DATE)
    assert time.recorded_sleeps == []


def test_calls_to_chart_data_do_not_exceed_max_calls_per_second(requests, random, time, repeater):
    random.last_random_value = milliseconds(10)
    repeater.function = lambda: return_chart_data("CASH", "SYMBOL", 300, A_UNIX_DATE)

    repeater.repeat_in(range(0, sut.MAX_CALLS_PER_SECOND), 0.1)

    time.set(sut.MAX_CALLS_PER_SECOND * 0.1)
    return_chart_data("CASH", "SYMBOL", 300, A_UNIX_DATE)

    repeater.repeat_in(range(sut.MAX_CALLS_PER_SECOND + 1, sut.MAX_CALLS_PER_SECOND * 2), 0.1)

    assert time.recorded_sleeps == [0.6]

    time.set(sut.MAX_CALLS_PER_SECOND * 2 * 0.1)
    return_chart_data("CASH", "SYMBOL", 300, A_UNIX_DATE)

    assert time.recorded_sleeps == [0.6, 0.6]
