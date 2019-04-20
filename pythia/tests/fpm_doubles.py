from collections.__init__ import OrderedDict


class Prices:
    def __init__(self, rates):
        self.rates = OrderedDict(rates)

    def to_array(self):
        t = []
        for v in self.rates.values():
            t.append([v['close'], v['high'], v['low']])
        return t


class TimeStub:
    def __init__(self, t):
        self.t = t

    def set(self, t):
        self.t = t

    def time(self, *args, **kwargs):
        return self.t


class TimeSpy(TimeStub):
    def __init__(self, t):
        super().__init__(t)
        self.sleeps = 0
        self.recorded_sleeps = list()

    def sleep(self, seconds):
        self.sleeps = seconds
        self.recorded_sleeps.append(seconds)


class RandomStub:
    def __init__(self):
        self.last_random_value = 42

    def randint(self, start, end):
        return self.last_random_value
