import json

from decimal import Decimal


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
    def __init__(self, stream):
        """
        Thin wrapper around a shape shift coin exchange market info json file. Provides simple iterator mechanics to
        walk efficiently through the coin exchange stream. Random access is not provided and stream has to be reset.

        :param stream: Steam of data containing a new line separated list of shapeshift market info json strings
        """
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

    def reset(self):
        self.stream.seek(0)