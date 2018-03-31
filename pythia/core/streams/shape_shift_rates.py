import json

from decimal import Decimal
from functools import reduce
from io import SEEK_CUR
from math import sqrt

SUPPORTED_COINS = [
    "BTC",
    "1ST",
    "ANT",
    "BAT",
    "BNT",
    "BCH",
    "BTG",
    "BLK",
    "CVC",
    "CLAM",
    "DASH",
    "DCR",
    "DGB",
    "DNT",
    "DOGE",
    "EDG",
    "EOS",
    "ETH",
    "ETC",
    "FCT",
    "GAME",
    "GNO",
    "GNT",
    "GUP",
    "KMD",
    "LBC",
    "LTC",
    "MONA",
    "NEO",
    "NMC",
    "XEM",
    "NMR",
    "NXT",
    "OMG",
    "POT",
    "REP",
    "RDD",
    "RCN",
    "RLC",
    "SALT",
    "SC",
    "SNT",
    "STORJ",
    "START",
    "SWT",
    "TRST",
    "VOX",
    "VRC",
    "VTC",
    "WAVES",
    "WINGS",
    "XMR",
    "XRP",
    "ZEC",
    "ZRX"
]


class RatesPair:
    def __init__(self, info):
        self._pair = info["pair"]
        self.rate = Decimal(info["rate"])
        self.limit = float(info["limit"])
        self.maxLimit = float(info["maxLimit"])
        self.min = float(info["min"])
        self.minerFee = Decimal(str(info["minerFee"]))

    def __str__(self):
        return '{{"rate":"{}","limit":{},"pair":"{}","maxLimit":{},"min":{},"minerFee":{}}}' \
            .format(self.rate, self.limit, self._pair, self.maxLimit, self.min, self.minerFee)

    def __repr__(self):
        return 'RatesPair: {}'.format(str(self))


class ShapeShiftRates:
    def __init__(self, stream, preload=False):
        """
        Thin wrapper around a shape shift coin exchange market info json file. Provides simple iterator mechanics to
        walk efficiently through the coin exchange stream. Random access is not provided and stream has to be reset.

        :param stream: Steam of data containing a new line separated list of shapeshift market info json strings
        :param preload: Optionally load the rates data into memory for faster access
        """
        self.stream = stream
        self.preload = preload
        self.cache = None
        self.cache_idx = 0
        if self.preload:
            self.cache = [p for p in self]
            self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        if self.cache is not None:
            return self._return_from_cache()

        line = self.stream.readline()
        if line == "":
            raise StopIteration

        json_obj = json.loads(line[line.find('['):line.find(']') + 1])
        pairs = dict()
        for pair in json_obj:
            pairs[pair["pair"]] = RatesPair(pair)

        return pairs

    def _return_from_cache(self):
        self.cache_idx += 1
        if self.cache_idx > len(self.cache):
            raise StopIteration
        return self.cache[self.cache_idx - 1]

    def reset(self):
        self.stream.seek(0)
        self.cache_idx = 0


def rates_filter(in_stream, exchanges):
    def concat_rates(rates_str, pairs):
        f = {k: pairs[k] for k in exchanges if k in pairs} if exchanges != [] else pairs
        if bool(f) is False:
            return rates_str

        return rates_str + "[ " + ",".join(map(str, f.values())) + "]\n"

    return reduce(concat_rates, in_stream, "")


class InterimLookahead:
    def __init__(self, rates):
        self.rates = rates
        self.cache_idx = None
        self.seek_offset = None

    def __enter__(self):
        if self.rates.cache is not None:
            self.cache_idx = self.rates.cache_idx
        else:
            self.seek_offset = self.rates.stream.seek(0, SEEK_CUR)

        return self.rates

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache_idx is not None:
            self.rates.cache_idx = self.cache_idx
        else:
            self.rates.stream.seek(self.seek_offset)


def interim_lookahead(rates):
    return InterimLookahead(rates)


ANALYSIS_STR_HEADER = " EXCHANGE |   MEAN   |    SD    |  MEDIAN  |   MIN   |   MAX   |   DIF   \n" \
                      "-------------------------------------------------------------------------\n"

ANALYSIS_CSV_HEADER = "EXCHANGE,MEAN,SD,MEDIAN,MIN,MAX,DIF\n"


class CoinExchangeReport:
    def __init__(self):
        self.records = list()

    def append(self, *columns):
        self.records.append(columns)

    def __str__(self):
        if len(self.records) == 0:
            return ""

        def print_records(report, columns):
            exchange, mean, sd, median, min, max, dif = columns
            return report + " {:<9}|{:>10.10}|{:>10.10}|{:>10.10}|{:>9.9}|{:>9.9}|{:>9.9}\n" \
                .format(exchange, str(mean), str(sd), str(median), str(min), str(max), str(dif))

        return reduce(print_records, self.records, ANALYSIS_STR_HEADER)

    def to_csv(self):
        if len(self.records) == 0:
            return ""

        def print_records(report, columns):
            return report + ",".join(map(str, columns)) + "\n"

        return reduce(print_records, self.records, ANALYSIS_CSV_HEADER)


def analyze(rates):
    def fold_rates(all, pairs):
        for key in pairs:
            if key not in all:
                all[key] = list()
            all[key].append(pairs[key].rate)
        return all

    exchanges = reduce(fold_rates, rates, dict())

    report = CoinExchangeReport()
    for k in exchanges:
        r = exchanges[k]
        dif = r[-1] - r[0]
        r = sorted(r)
        s = sum(r)
        l = len(r)
        mean = s / l
        median = r[(l // 2)]
        sd = sqrt(reduce(lambda t, x: t + (x - mean) ** 2, r, Decimal(0)) / l)
        report.append(k, mean, sd, median, min(r), max(r), dif)

    return report


class ExchangeRanges:
    def __init__(self):
        self.ranges = dict()

    def __getitem__(self, pair_name):
        return self.ranges[pair_name]

    def extend(self, pair):
        if pair.name not in self.ranges:
            self.ranges[pair.name] = pair
        else:
            self.ranges[pair.name].extend(pair)

        return self

    def normalize_rate(self, name, rate):
        r = self.ranges[name]
        if (r.max - r.min) == 0:
            return 0

        return (float(rate) - r.min) / (r.max - r.min)

    def __len__(self):
        return len(self.ranges)


class PairRange:
    def __init__(self, index, named_pair):
        self.min_position = index
        self.max_position = index
        name, pair = named_pair
        self.name = name
        self.min = float(pair.rate)
        self.max = float(pair.rate)

    def extend(self, other):
        assert self.name == other.name
        if other.min < self.min:
            self._set_minimum(other)
        if other.max > self.max:
            self._set_maximum(other)
        return self

    def _set_maximum(self, other):
        self.max = other.max
        self.max_position = other.max_position

    def _set_minimum(self, other):
        self.min = other.min
        self.min_position = other.min_position


class PairMaxDifference(PairRange):
    def __init__(self, index, named_pair, target_diff):
        super().__init__(index, named_pair)
        self.target_diff = target_diff
        self.max_difference = 0
        self.start_position = 0
        self.end_position = 0

    def extend(self, other):
        if self.target_diff is not None and self.max_difference >= self.target_diff:
            return
        super().extend(other)

    def _set_minimum(self, other):
        super()._set_minimum(other)
        d = (self.max - self.min) / self.max
        if d > self.max_difference:
            self.max_difference = d
            self.end_position = self.min_position
            self.start_position = self.max_position

    def _set_maximum(self, other):
        super()._set_maximum(other)
        self.min = self.max
        self.min_position = self.max_position


class PairMaxDiffFactory:
    def __init__(self, target_diff):
        self.target_diff = target_diff

    def __call__(self, index, named_pair):
        return PairMaxDifference(index, named_pair, self.target_diff)


def _pair_to_accumulator(idx_pairs, accumulator):
    idx, pairs = idx_pairs
    return map(lambda pair: accumulator(idx, pair), pairs.items())


def _reduce_to_pairs_to_accumulator(rates, accumulator):
    def fold_to_accumulator(acc, pair):
        return acc.extend(pair)

    def extend_exchange_ranges(ranges, idx_pairs):
        return reduce(fold_to_accumulator, _pair_to_accumulator(idx_pairs, accumulator), ranges)

    return reduce(extend_exchange_ranges, enumerate(rates), ExchangeRanges())


def calculate_exchange_ranges(rates):
    return _reduce_to_pairs_to_accumulator(rates, PairRange)


def calculate_exchange_max_differences(rates, target_diff=None):
    maxDiff = PairMaxDiffFactory(target_diff)
    return _reduce_to_pairs_to_accumulator(rates, maxDiff)
