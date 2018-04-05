import csv
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


class Symbol:
    def __init__(self, name, stream):
        self.name = name
        self.stream = stream
        self.csv_reader = csv.reader(self.stream)

    def __iter__(self):
        return self

    def __next__(self):
        columns = next(self.csv_reader)
        if self._is_header(columns):
            columns = next(self.csv_reader)

        return tuple(map(float, columns[1:5])) + tuple([int(columns[5])])

    @staticmethod
    def _is_header(columns):
        return columns[0] == "timestamp"

    def seek(self, *args, **kwargs):
        return self.stream.seek(*args, **kwargs)


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
        self.symbol_stream.seek(0)
        if self.second_symbol_stream is not None:
            self.second_symbol_stream.seek(0)


class InterimLookahead:
    def __init__(self, rates):
        self.rates = rates
        self.offset_a = None
        self.offset_b = None

    def __enter__(self):
        self.offset_a = self.rates.symbol_stream.seek(0, SEEK_CUR)
        if self.rates.second_symbol_stream is not None:
            self.offset_b = self.rates.second_symbol_stream.seek(0, SEEK_CUR)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rates.symbol_stream.seek(self.offset_a)
        if self.offset_b is not None:
            self.rates.second_symbol_stream.seek(self.offset_b)


def interim_lookahead(rates):
    return InterimLookahead(rates)