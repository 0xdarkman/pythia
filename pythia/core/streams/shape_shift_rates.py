import json

from decimal import Decimal
from functools import reduce
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


ANALYSIS_STR_HEADER = " EXCHANGE |   MEAN   |    SD    |  MEDIAN  |   MIN   |   MAX   \n" \
                      "---------------------------------------------------------------\n"


ANALYSIS_CSV_HEADER = "EXCHANGE,MEAN,SD,MEDIAN,MIN,MAX\n"

class CoinExchangeReport:
    def __init__(self):
        self.records = list()

    def append(self, *columns):
        self.records.append(columns)

    def __str__(self):
        if len(self.records) == 0:
            return ""

        def print_records(report, columns):
            exchange, mean, sd, median, min, max = columns
            return report + " {:<9}|{:>10.10}|{:>10.10}|{:>10.10}|{:>9.9}|{:>9.9}\n" \
                .format(exchange, str(mean), str(sd), str(median), str(min), str(max))

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
        r = sorted(exchanges[k])
        s = sum(r)
        l = len(r)
        mean = s / l
        median = r[(l // 2)]
        sd = sqrt(reduce(lambda t, x: t + (x - mean)**2, r, Decimal(0)) / l)
        report.append(k, mean, sd, median, min(r), max(r))

    return report