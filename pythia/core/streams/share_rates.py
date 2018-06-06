import csv
from decimal import Decimal
from io import SEEK_CUR


class RatesPair:
    def __init__(self, open, high, low, close, volume):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __str__(self):
        return "{{open={}, high={}, low={}, close={}, volume={}}}" \
            .format(self.open, self.high, self.low, self.close, self.volume)

    def __repr__(self):
        return "RatesPair: {}".format(str(self))

    @property
    def rate(self):
        return Decimal(self.open)

    @property
    def fee(self):
        return 0


class Symbol:
    def __init__(self, name, stream):
        self.name = name
        self.stream = stream
        self.csv_reader = csv.reader(self.stream)
        self.rows = [tuple(map(float, c[1:5])) + tuple([int(c[5])]) for c in self.csv_reader if not self._is_header(c)]
        self.idx = 0
        self.size = len(self.rows)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == self.size:
            raise StopIteration

        r = self.rows[self.idx]
        self.idx += 1
        return r

    @staticmethod
    def _is_header(columns):
        return columns[0] == "timestamp"

    def seek(self, *args, **kwargs):
        return self.stream.seek(*args, **kwargs)

    def reset(self):
        self.idx = 0


class IdentitySymbol:
    def __iter__(self):
        return self


class ShareRates:
    FIAT_TAG = "CURRENCY"

    def __init__(self, symbol_stream, second_symbol_stream=None):
        self.symbol_stream = symbol_stream
        self.second_symbol_stream = second_symbol_stream

    def __iter__(self):
        return self

    def __next__(self):
        symbol_a, name_a = next(self.symbol_stream), self.symbol_stream.name
        symbol_b, name_b = self._get_other_symbol(symbol_a[4])

        def calc_rate(t):
            a, b = t
            return a / b

        rates = dict()
        rates[name_a + "_" + name_b] = RatesPair(
            *map(calc_rate, zip(symbol_a[0:4], symbol_b[0:4])), symbol_a[4])
        rates[name_b + "_" + name_a] = RatesPair(
            *map(calc_rate, zip(symbol_b[0:4], symbol_a[0:4])), symbol_b[4])
        return rates

    def _get_other_symbol(self, volume_a):
        if self.second_symbol_stream is not None:
            return next(self.second_symbol_stream), self.second_symbol_stream.name
        else:
            return (1, 1, 1, 1, volume_a), self.FIAT_TAG

    def reset(self):
        self.symbol_stream.reset()
        if self.second_symbol_stream is not None:
            self.second_symbol_stream.reset()

    def lookahead(self):
        return InterimLookahead(self)


class InterimLookahead:
    def __init__(self, rates):
        self.rates = rates
        self.offset_a = None
        self.offset_b = None

    def __enter__(self):
        self.offset_a = self.rates.symbol_stream.idx
        if self.rates.second_symbol_stream is not None:
            self.offset_b = self.rates.second_symbol_stream.idx
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rates.symbol_stream.idx = self.offset_a
        if self.offset_b is not None:
            self.rates.second_symbol_stream.idx = self.offset_b
