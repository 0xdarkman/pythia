import random
import time
from collections import deque

import requests

from pythia.core.remote.poloniex_api import return_chart_data

RANDOM_DELAY = (3, 5)


def _retry_timeout(f, n):
    for _ in range(0, n):
        try:
            return f()
        except requests.exceptions.RequestException:
            pass

    raise PoloniexConnection.TimeoutError("{} timed out after {} retries".format(str(f), n))


def _retry_delayed_if(p, f, n, info):
    r = None
    for _ in range(0, n):
        r = f()
        if p(r):
            time.sleep(random.randint(*RANDOM_DELAY))
        else:
            return r

    raise PoloniexConnection.NoDataError(
        "{} no valid data after {} retries\nResult: {}\n{}".format(str(f), n, str(r), info))


def _take_keys(d, *ks):
    return {k: d[k] for k in ks}


class PoloniexConnection:
    def __init__(self, telemetry, config):
        self._telemetry = telemetry
        self._period = config["period"]
        self._start_time = config["start"]
        self._retries = config["retry"]
        self._buffers = dict()

    def get_next_prices(self, cash, symbol):
        pair = "{}_{}".format(cash, symbol)
        next_ts = self._telemetry.find_last_chart_ts(pair, self._start_time) + self._period
        self._wait_for_interval(next_ts)

        raw = self._get_data_of(pair, cash, symbol, next_ts)
        chart = _take_keys(raw, "close", "high", "low", "date")
        self._telemetry.write_chart({pair: chart})
        return _take_keys(chart, "close", "high", "low")

    def _wait_for_interval(self, next_ts):
        cur_ts = time.time()
        period_complete = next_ts + self._period
        if cur_ts <= period_complete:
            time.sleep(period_complete - cur_ts + random.randint(*RANDOM_DELAY))

    def _get_data_of(self, pair, cash, symbol, start):
        if pair not in self._buffers or len(self._buffers[pair]) == 0:
            self._buffers[pair] = deque(self._request_charts(cash, symbol, start))

        return self._buffers[pair].popleft()

    def _request_charts(self, cash, symbol, start):

        def get_charts():
            def query_charts():
                end = self._calc_next_full_chart_ts(start)
                return return_chart_data(cash, symbol, self._period, start, end)

            return _retry_timeout(query_charts, self._retries)

        info = "{}: {}_{} start: {}, end: {}".format(time.time(), cash, symbol, start,
                                                     self._calc_next_full_chart_ts(start))
        return _retry_delayed_if(self._is_empty_chart, get_charts, self._retries, info)

    def _calc_next_full_chart_ts(self, start):
        current = time.time()
        return start + max(int((current - start) / self._period) - 1, 0) * self._period

    @staticmethod
    def _is_empty_chart(charts):
        if len(charts) > 1:
            return False
        c = charts[0]
        return all([c[k] == 0 for k in c])

    def reset(self):
        self._telemetry.reset()

    class TimeoutError(TimeoutError):
        pass

    class NoDataError(RuntimeError):
        pass
