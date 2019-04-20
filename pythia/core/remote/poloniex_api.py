import random
import time

import requests

PUBLIC_API = "https://poloniex.com/public"
CMD_CHART = "returnChartData"

MAX_CALLS_PER_SECOND = 4
VALID_PERIODS = [300, 900, 1800, 7200, 14400, 86400]
RANDOM_DELAY_INTERVAL_IN_MS = (170, 200)

_previous_request_times = []


def _ms_to_sec(ms):
    return ms * 0.001


class WaitTimeUpdate:
    def __init__(self):
        self._now = None

    def __enter__(self):
        self._now = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _previous_request_times.append(self._now)
        pass

    def calc_remaining(self):
        num_requests = len(_previous_request_times)
        if num_requests == 0:
            return 0

        if num_requests == MAX_CALLS_PER_SECOND:
            r = 1.0 - (self._now - _previous_request_times[0])
            _previous_request_times.clear()
            return r

        delay = _ms_to_sec(random.randint(*RANDOM_DELAY_INTERVAL_IN_MS))
        return delay - (self._now - _previous_request_times[-1])


def _delayed_get(*args, **kwargs):
    with WaitTimeUpdate() as wait_time:
        remaining = wait_time.calc_remaining()
        if remaining > 0:
            time.sleep(remaining)

    return requests.get(*args, **kwargs)


def return_chart_data(cash, symbol, period, start, end=None):
    if period not in VALID_PERIODS:
        raise InvalidParameterError("Period {} is not valid. Valid periods are: {}".format(period, VALID_PERIODS))

    result = _delayed_get(PUBLIC_API, _make_chart_payload(cash, symbol, period, start, end), timeout=3.05)
    result.raise_for_status()
    return result.json()


def _make_chart_payload(cash, symbol, period, start, end):
    base = {"command": CMD_CHART, "currencyPair": "{}_{}".format(cash, symbol), "period": period, "start": start}
    if end is not None:
        base["end"] = end
    return base


class InvalidParameterError(ValueError):
    pass
