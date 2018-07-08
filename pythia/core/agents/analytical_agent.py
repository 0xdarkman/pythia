from decimal import Decimal
from functools import reduce


class AnalyticalAgent:
    def __init__(self, min_distance, diff_threshold, diff_window, targets):
        """
        Analytical agent is exchanging coins depending on a normalized minimum distance and a differential threshold

        :param min_distance: (Decimal) the normalized minimum relative distance between two token exchange rates necessary
        :param diff_threshold: (Decimal) the minimum normalized differential for an exchange
        :param diff_window: (Integer) the moving window that is used to calculate the differential of the exchange rate
        """
        self.min_distance = Decimal(min_distance)
        self.differential_threshold = Decimal(diff_threshold)
        self.differential_window = diff_window
        self.targets = targets

        self._rate_recordings = dict()
        self._current_coin = None

    def start(self, state):
        return self.step(state)

    def step(self, state, reward=None):  # rewards are not used by this agent
        self._current_coin, markets = state
        for target in self.targets:
            market = self._get_market_info_of(markets, target)
            if market.rate == Decimal('0'):
                continue

            self._record_for(market, target)
            if self._should_exchange(market, target):
                return self._do_exchange(markets, target)

        return None

    def _get_market_info_of(self, rates, target):
        exchange = self._current_coin + "_" + target
        rate_info = rates[exchange]
        return rate_info

    def _record_for(self, market, target):
        if target not in self._rate_recordings:
            self._rate_recordings[target] = self._Recording(market.rate)
        self._rate_recordings[target].differentials.append(market.rate)

    def _should_exchange(self, rates_info, target):
        rec = self._rate_recordings[target]
        if len(rec.differentials) == self.differential_window:
            distance = ((rates_info.rate - rates_info.fee) - rec.initial_rate) / rec.initial_rate
            differential = self._calc_differential(target)
            return distance >= self.min_distance and differential >= self.differential_threshold

        return False

    def _calc_differential(self, target):
        def calc_diff(dist, rec_touple):
            prev, cur = rec_touple
            return dist + ((prev - cur) / prev)

        rec = self._rate_recordings[target]
        next_rec = rec.differentials[1:]
        cur_rec = rec.differentials[:-1]
        self._rate_recordings[target].differentials = next_rec
        return reduce(calc_diff, zip(cur_rec, next_rec), 0.0) / len(next_rec)

    def _do_exchange(self, rates, target):
        self.targets.remove(target)
        self.targets.append(self._current_coin)

        self._rate_recordings.clear()
        for new_t in self.targets:
            self._record_for(rates[target + "_" + new_t], new_t)
        return target

    def finish(self, reward):
        pass

    class _Recording:
        def __init__(self, init_rate):
            self.initial_rate = init_rate
            self.differentials = list()
