from decimal import Decimal
from functools import reduce


class AnalyticalAgent:
    def __init__(self, min_distance, diff_threshold, diff_window):
        self.min_distance = Decimal(min_distance)
        self.differential_threshold = Decimal(diff_threshold)
        self.differential_window = diff_window
        self.differential_records = list()
        self.start_rate = None
        self.exchange = "BTC_ETH"

    def step(self, rates):
        rate_info = rates[self.exchange]
        self._record(rate_info)
        if self._should_exchange(rate_info):
            return self._do_exchange(rates)

        return None

    def _record(self, rate_info):
        if self.start_rate is None:
            self.start_rate = rate_info.rate
        self.differential_records.append(rate_info.rate)

    def _should_exchange(self, rates_info):
        if len(self.differential_records) == self.differential_window:
            distance = ((rates_info.rate - rates_info.minerFee) - self.start_rate) / self.start_rate
            differential = self._calc_differential()
            return distance >= self.min_distance and differential >= self.differential_threshold

        return False

    def _calc_differential(self):
        def calc_diff(dist, rec_touple):
            prev, cur = rec_touple
            return dist + ((prev - cur) / prev)

        next_rec = self.differential_records[1:]
        cur_rec = self.differential_records[:-1]
        self.differential_records = next_rec
        return reduce(calc_diff, zip(cur_rec, next_rec), Decimal(0)) / len(self.differential_records)

    def _do_exchange(self, rates):
        e = self.exchange
        self.exchange = "ETH_BTC"

        self.start_rate = None
        self.differential_records.clear()
        self._record(rates[self.exchange])
        return e