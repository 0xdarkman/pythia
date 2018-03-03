import io

from pythia.core.reinforcement.q_table import QTable
from pythia.core.streams.shape_shift_rates import RatesPair, ShapeShiftRates


class QFunctionWrapper(QTable):
    def set_state_action_values(self, state, zero_value, one_value):
        self[state, 0] = zero_value
        self[state, 1] = one_value


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


class RecordsStub(io.StringIO):
    def add_record(self, *pairs):
        string = "{2018-02-23 07:55:01.668919: b'[" + ",".join(map(str, pairs)) + "]'}\n"
        self.write(string)
        return self

    def finish(self):
        self.seek(0)


class RatesStub(ShapeShiftRates):
    def __init__(self, stream):
        super().__init__(stream)
        self.stream = stream

    def add_record(self, *pairs):
        self.stream.add_record(*pairs)
        return self

    def finish(self):
        self.stream.finish()

    def close(self):
        self.stream.close()


def entry(pair, rate, miner_fee="0.5"):
    return PairEntryStub(pair, rate, "0.7", "6.2", "0.1", miner_fee)