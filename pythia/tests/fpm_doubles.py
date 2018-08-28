from collections.__init__ import OrderedDict


class Prices:
    def __init__(self, rates):
        self.rates = OrderedDict(rates)

    def to_array(self):
        t = []
        for v in self.rates.values():
            t.append([v['close'], v['high'], v['low']])
        return t
