import json


class RatesPair:
    def __init__(self, info):
        self._pair = info["pair"]
        self.rate = float(info["rate"])
        self.limit = float(info["limit"])
        self.maxLimit = float(info["maxLimit"])
        self.min = float(info["min"])
        self.minerFee = float(info["minerFee"])

    def __str__(self):
        return '{{"rate":"{}","limit":{},"pair":"{}","maxLimit":{},"min":{},"minerFee":{}}}' \
            .format(self.rate, self.limit, self._pair, self.maxLimit, self.min, self.minerFee)

    def __repr__(self):
        return 'RatesPair: {}'.format(str(self))


class ShapeShiftRates:
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        line = self.stream.readline()
        if line == "":
            raise StopIteration

        json_obj = json.loads(line[line.find('['):line.find(']') + 1])
        pairs = dict()
        for pair in json_obj:
            pairs[pair["pair"]] = RatesPair(pair)

        return pairs