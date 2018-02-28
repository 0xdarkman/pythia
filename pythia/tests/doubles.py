from pythia.reinforcement.q_table import QTable
from pythia.streams.shape_shift_rates import RatesPair


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