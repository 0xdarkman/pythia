from functools import reduce


def rates_filter(in_stream, exchanges):
    def concat_rates(rates_str, pairs):
        f = {k: pairs[k] for k in exchanges if k in pairs} if exchanges != [] else pairs
        if bool(f) is False:
            return rates_str

        return rates_str + "[ " + ",".join(map(str, f.values())) + "]\n"

    return reduce(concat_rates, in_stream, "")


class ExchangeRanges:
    def __init__(self):
        self.ranges = dict()

    def __getitem__(self, pair_name):
        return self.ranges[pair_name]

    def extend(self, pair):
        if pair.name not in self.ranges:
            self.ranges[pair.name] = pair
        else:
            self.ranges[pair.name].extend(pair)

        return self

    def normalize_rate(self, name, rate):
        r = self.ranges[name]
        if (r.max - r.min) == 0:
            return 0

        return (float(rate) - r.min) / (r.max - r.min)

    def __len__(self):
        return len(self.ranges)


class PairRange:
    def __init__(self, index, named_pair):
        self.min_position = index
        self.max_position = index
        name, pair = named_pair
        self.name = name
        self.min = float(pair.rate)
        self.max = float(pair.rate)

    def extend(self, other):
        assert self.name == other.name
        if other.min < self.min:
            self._set_minimum(other)
        if other.max > self.max:
            self._set_maximum(other)
        return self

    def _set_maximum(self, other):
        self.max = other.max
        self.max_position = other.max_position

    def _set_minimum(self, other):
        self.min = other.min
        self.min_position = other.min_position


class PairMaxDifference(PairRange):
    def __init__(self, index, named_pair, target_diff):
        super().__init__(index, named_pair)
        self.target_diff = target_diff
        self.max_difference = 0
        self.start_position = 0
        self.end_position = 0

    def extend(self, other):
        if self.target_diff is not None and self.max_difference >= self.target_diff:
            return
        super().extend(other)

    def _set_minimum(self, other):
        super()._set_minimum(other)
        d = (self.max - self.min) / self.max
        if d > self.max_difference:
            self.max_difference = d
            self.end_position = self.min_position
            self.start_position = self.max_position

    def _set_maximum(self, other):
        super()._set_maximum(other)
        self.min = self.max
        self.min_position = self.max_position


class PairMaxDiffFactory:
    def __init__(self, target_diff):
        self.target_diff = target_diff

    def __call__(self, index, named_pair):
        return PairMaxDifference(index, named_pair, self.target_diff)


def _pair_to_accumulator(idx_pairs, accumulator):
    idx, pairs = idx_pairs
    return map(lambda pair: accumulator(idx, pair), pairs.items())


def _reduce_to_pairs_to_accumulator(rates, accumulator):
    def fold_to_accumulator(acc, pair):
        return acc.extend(pair)

    def extend_exchange_ranges(ranges, idx_pairs):
        return reduce(fold_to_accumulator, _pair_to_accumulator(idx_pairs, accumulator), ranges)

    return reduce(extend_exchange_ranges, enumerate(rates), ExchangeRanges())


def calculate_exchange_ranges(rates):
    return _reduce_to_pairs_to_accumulator(rates, PairRange)


def calculate_exchange_max_differences(rates, target_diff=None):
    maxDiff = PairMaxDiffFactory(target_diff)
    return _reduce_to_pairs_to_accumulator(rates, maxDiff)