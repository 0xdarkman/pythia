import io
import json

import pytest


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


class PairEntryStub(RatesPair):
    def __init__(self, pair, rate, limit, maxLimit, min, minerFee):
        super().__init__({"pair": pair,
                          "rate": rate,
                          "limit": limit,
                          "maxLimit": maxLimit,
                          "min": min,
                          "minerFee": minerFee})

    def __eq__(self, other):
        return self.rate == other.rate and \
               self.limit == other.limit and \
               self.maxLimit == other.maxLimit and \
               self.min == other.min and \
               self.minerFee == other.minerFee


@pytest.fixture
def empty():
    stream = io.StringIO("")
    yield ShapeShiftRates(stream)
    stream.close()


def test_empty(empty):
    with pytest.raises(StopIteration) as e:
        unused = next(iter(empty))


def test_one_pair_entry():
    pair = PairEntryStub("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)
    with io.StringIO('{2018-02-23 07:55:01.668919: b\'[' + str(pair) + ']\'}') as stream:
        single = ShapeShiftRates(stream)
        pairs = next(iter(single))
        assert pairs["ETH_SALT"] == pair


def test_multiple_pair_entries():
    pair1 = PairEntryStub("ETH_SALT", 1.1, 0.7, 6.2, 0.1, 0.5)
    pair2 = PairEntryStub("RCN_1ST", 1.52, 0.1, 4.1, 1.1, 1.9)
    with io.StringIO('{2018-02-23 07:55:01.668919: b\'[' + str(pair1) + ', ' + str(pair2) + ']\']\'}') as stream:
        single = ShapeShiftRates(stream)
        pairs = next(iter(single))
        assert pairs["ETH_SALT"] == pair1
        assert pairs["RCN_1ST"] == pair2
