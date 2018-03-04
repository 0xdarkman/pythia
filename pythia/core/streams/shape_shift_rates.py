import json

from decimal import Decimal
from functools import reduce

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
