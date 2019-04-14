import random
import time
from collections import deque

from pythia.core.remote.poloniex_api import return_chart_data

RANDOM_DELAY = (30, 120)


def _retry_timeout(f, n):
    for _ in range(0, n):
        try:
            return f()
        except TimeoutError:
            pass

    raise PoloniexConnection.TimeoutError("{} timed out after {} retries".format(str(f), n))


def _retry_delayed_if(p, f, n):
    for _ in range(0, n):
        r = f()
        if p(r):
            time.sleep(random.randint(*RANDOM_DELAY))
        else:
            return r

    raise PoloniexConnection.NoDataError("{} no valid data after {} retries".format(str(f), n))


def _take_keys(d, *ks):
    return {k: d[k] for k in ks}


class PoloniexConnection:
    def __init__(self, telemetry, config):
        self._telemetry = telemetry
        self._period = config["period"]
        self._start_time = config["start"]
        self._retries = config["retry"]
        self._buffer = deque()

    def get_next_prices(self, cash, symbol):
        next_ts = self._telemetry.find_last_chart_ts(self._start_time) + self._period
        self._wait_for_interval(next_ts)

        if len(self._buffer) == 0:
            self._buffer = deque(self._request_charts(cash, symbol, next_ts))

        chart = _take_keys(self._buffer.popleft(), "close", "high", "low", "date")
        self._telemetry.write_chart({"{}_{}".format(cash, symbol): chart})
        return _take_keys(chart, "close", "high", "low")

    def _request_charts(self, cash, symbol, ts):
        def get_charts():
            def query_charts():
                return return_chart_data(cash, symbol, self._period, ts)

            return _retry_timeout(query_charts, self._retries)

        return _retry_delayed_if(self._is_empty_chart, get_charts, self._retries)

    @staticmethod
    def _wait_for_interval(next_ts):
        cur_ts = time.time()
        if cur_ts <= next_ts:
            time.sleep(next_ts - cur_ts + random.randint(*RANDOM_DELAY))

    @staticmethod
    def _is_empty_chart(charts):
        if len(charts) > 1:
            return False
        c = charts[0]
        return all([c[k] == 0 for k in c])

    class TimeoutError(TimeoutError):
        pass

    class NoDataError(RuntimeError):
        pass
