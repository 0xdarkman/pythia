from decimal import Decimal

CASH_TOKEN = "CASH"


class DataFrameStream:
    def __init__(self, frame, name):
        self.frame = frame
        self.iterator = self.frame.iterrows()
        self.name = name

    def __iter__(self):
        return self

    def __next__(self):
        r = next(self.iterator)[1]
        return {"{}_{}".format(self.name, CASH_TOKEN): Exchange(1.0 / r.open, 1.0 / r.high, 1.0 / r.low, 1.0 / r.close,
                                                                r.volume),
                "{}_{}".format(CASH_TOKEN, self.name): Exchange(r.open, r.high, r.low, r.close, r.volume)}

    def reset(self):
        self.iterator = self.frame.iterrows()


class Exchange:
    def __init__(self, open, high, low, close, volume):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.rate = self.close
        self.fee = 0.0

    def __eq__(self, other):
        return self.open == other.open and \
               self.high == other.high and \
               self.low == other.low and \
               self.close == other.close and \
               self.volume == other.volume

    def __str__(self):
        return "{{open={}, high={}, low={}, close={}, volume={}}}" \
            .format(self.open, self.high, self.low, self.close, self.volume)

    def __repr__(self):
        return "RatesPair: {}".format(str(self))
