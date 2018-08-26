import time
from collections import OrderedDict

import numpy as np
import pytest

from pythia.core.agents.fpm_memory import FPMMemory


class Prices:
    def __init__(self, rates):
        self.rates = OrderedDict(rates)

    def to_array(self):
        t = []
        for v in self.rates.values():
            t.append([v['close'], v['high'], v['low']])
        return t


class MemoryTestBuilder:
    def __init__(self):
        self.window = 1
        self.beta = 0.1
        self._memory = None

    def record(self, prices, portfolio):
        self.memory.record(prices.to_array(), portfolio)

    def get_latest(self):
        return self.memory.get_latest()

    def get_random_batch(self, size):
        return self.memory.get_random_batch(size)

    def update(self, batch):
        return self.memory.update(batch)

    def ready(self):
        return self.memory.ready()

    @property
    def memory(self):
        if self._memory is None:
            cfg = {"training": {"window": self.window, "size": 1000, "beta": self.beta}}
            self._memory = FPMMemory(cfg)
        return self._memory


@pytest.fixture
def memory():
    return MemoryTestBuilder()


def environment_input(identifier):
    w0 = 1.0 / identifier
    return Prices({"SYM": {"high": 1.0, "low": 1.0, "close": identifier}}), [w0, 1 - w0]


def state(identifier):
    w0 = 1.0 / identifier
    return [[[1.0]], [[1.0 / identifier]], [[1.0 / identifier]]], [w0, 1 - w0]


def identify_state(state):
    prices, _, _ = state
    return int(1.0 / prices[1][0][0])


def batch(*identifiers):
    prices, weights, future = [], [], []
    for i in range(0, len(identifiers) - 1):
        ident = identifiers[i]
        p, w = state(ident)
        prices.append(p)
        weights.append(w)
        future.append(identifiers[i + 1] / ident)

    return FPMMemory.Batch(np.array(prices), np.array(weights), np.array(future), 0, len(identifiers))


def record(memory, size):
    for i in range(1, size + 1):
        memory.record(*environment_input(i))


def get_stable_batch(memory, size, seed):
    np.random.seed(seed)
    return memory.get_random_batch(size)


def assert_states(expected_prices, expected_portfolio, actual_prices, actual_portfolio):
    assert (np.array(expected_prices) == actual_prices).all()
    assert (np.array(expected_portfolio) == actual_portfolio).all()


def assert_weights(expected, actual):
    assert (np.array(expected) == actual).all()


def assert_batch(expected, actual):
    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        ep, ew, ef = e
        ap, aw, af = a
        assert (ep == ap).all()
        assert (ew == aw).all()
        assert (ef == af).all()


def test_nothing_recorded_latest_is_none(memory):
    pr, po = memory.get_latest()
    assert pr is None and po is None


@pytest.mark.parametrize("price, portfolio, price_tensor", [
    (Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 2.0}}), [1.0, 0.0], [[[1.0]], [[2.0]], [[0.5]]]),
    (Prices({"SYM2": {"high": 2.0, "low": 0.1, "close": 0.5}}), [0.5, 0.5], [[[1.0]], [[4.0]], [[0.2]]]),
])
def test_enough_records_latest_returns_last_price_tensor_and_portfolio(memory, price, portfolio, price_tensor):
    memory.record(price, portfolio)
    assert_states(price_tensor, portfolio, *memory.get_latest())


def test_multiple_assets(memory):
    p = Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 2.0}, "SYM2": {"high": 2.0, "low": 0.1, "close": 0.5}})
    memory.record(p, [1.0, 0.0, 0.0])
    assert_states([[[1.0], [1.0]], [[2.0], [4.0]], [[0.5], [0.2]]], [1.0, 0.0, 0.0], *memory.get_latest())


def test_raise_data_mismatch_error_when_amount_of_symbols_does_not_fit_portfolio_vector(memory):
    p = Prices({"SYM1": {"high": 2.0, "low": 1.0, "close": 1.5}, "SYM2": {"high": 1.0, "low": 0.1, "close": 0.8}})
    with pytest.raises(FPMMemory.DataMismatchError):
        memory.record(p, [1.0, 0.0])


def test_latest_returns_none_if_there_are_not_enough_records_to_fill_price_window(memory):
    memory.window = 2
    memory.record(Prices({"SYM1": {"high": 2.0, "low": 1.0, "close": 1.5}}), [1.0, 0.0])
    assert_states(None, None, *memory.get_latest())


def test_is_ready_when_enough_prices_are_recorded_to_fill_the_window(memory):
    memory.window = 2
    assert not memory.ready()
    memory.record(*environment_input(1))
    assert not memory.ready()
    memory.record(*environment_input(2))
    assert memory.ready()


def test_price_vector_contains_price_history_quotient_of_most_recent_price_in_window(memory):
    memory.window = 2
    memory.record(Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 1.5}}), [1.0, 0.0])
    memory.record(Prices({"SYM1": {"high": 2.5, "low": 1.2, "close": 3.0}}), [1.0, 0.0])
    assert_states([[[1.5 / 3.0, 3.0 / 3.0]], [[4.0 / 3.0, 2.5 / 3.0]], [[1.0 / 3.0, 1.2 / 3.0]]], [1.0, 0.0],
                  *memory.get_latest())


def test_window_with_multiple_assets(memory):
    memory.window = 2
    p1 = Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 1.0}, "SYM2": {"high": 2.0, "low": 0.1, "close": 1.0}})
    p2 = Prices({"SYM1": {"high": 2.0, "low": 0.5, "close": 2.0}, "SYM2": {"high": 3.0, "low": 0.2, "close": 0.5}})
    memory.record(p1, [1.0, 0.0, 0.0])
    memory.record(p2, [1.0, 0.0, 0.0])
    assert_states([[[0.5, 1.0], [2.0, 1.0]], [[2.0, 1.0], [4.0, 6.0]], [[0.5, 0.25], [0.2, 0.4]]], [1.0, 0.0, 0.0],
                  *memory.get_latest())


def test_drop_price_information_after_window(memory):
    memory.window = 2
    memory.record(Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 1.5}}), [1.0, 0.0])
    memory.record(Prices({"SYM1": {"high": 2.5, "low": 1.2, "close": 2.0}}), [1.0, 0.0])
    memory.record(Prices({"SYM1": {"high": 2.0, "low": 0.5, "close": 1.0}}), [1.0, 0.0])
    assert_states([[[2.0, 1.0]], [[2.5, 2.0]], [[1.2, 0.5]]], [1.0, 0.0], *memory.get_latest())


def test_return_empty_batch_when_nothing_is_recorded(memory):
    assert memory.get_random_batch(1).empty


def test_return_empty_batch_when_it_is_not_possible_to_provide_future_prices(memory):
    memory.record(*environment_input(1))
    assert memory.get_random_batch(1).empty


def test_return_trivial_random_batch(memory):
    memory.record(*environment_input(1.0))
    memory.record(*environment_input(2.0))
    assert_batch(batch(1.0, 2.0), memory.get_random_batch(1))


def test_select_randomly_from_history(memory):
    np.random.seed(11)
    memory.record(*environment_input(1.0))
    memory.record(*environment_input(2.0))
    memory.record(*environment_input(3.0))
    assert_batch(batch(1.0, 2.0), memory.get_random_batch(1))


def test_random_batches_preserve_time_series(memory):
    np.random.seed(7)
    memory.record(*environment_input(1.0))
    memory.record(*environment_input(2.0))
    memory.record(*environment_input(3.0))
    memory.record(*environment_input(4.0))
    assert_batch(batch(2.0, 3.0, 4.0), memory.get_random_batch(2))


def test_not_enough_data_to_fill_batch_size_truncates_the_batch(memory):
    np.random.seed(7)
    memory.record(*environment_input(1.0))
    memory.record(*environment_input(2.0))
    assert_batch(batch(1.0, 2.0), memory.get_random_batch(2))


def test_batch_selection_follows_a_geometrically_decaying_distribution(memory):
    np.random.seed(7)
    memory.beta = 0.5
    records = 6
    record(memory, records)

    distribution = [0] * (records - 2)
    n = 1000
    for _ in range(0, n):
        distribution[identify_state(memory.get_random_batch(2)[0]) - 1] += 1

    distribution[:] = [p / n for p in distribution]
    assert pytest.approx([0.125, 0.125, 0.25, 0.5], 0.1) == distribution


def test_portfolio_weights_of_a_batch_can_be_updated_with_predictions(memory):
    seed = int(time.time())  # should be stable in a random environment so select a random seed
    record(memory, 100)

    b = get_stable_batch(memory, 3, seed)
    old_w = list(b.weights)
    b.predictions = [[0.0, 0.0]] * 3
    memory.update(b)

    b = get_stable_batch(memory, 3, seed)
    assert_weights([old_w[0]] + [[0.0, 0.0]] * 2, b.weights)


def test_portfolio_weights_get_updated_by_predictions_up_to_one_after_the_batch(memory):
    record(memory, 4)
    b = get_stable_batch(memory, 2, 1)
    b.predictions = [[0.0, 0.0]] * 2
    memory.update(b)

    b = get_stable_batch(memory, 2, 7)
    assert_weights([[0.0, 0.0]] * 2, b.weights)


def test_portfolio_weight_update_is_clamped_to_record_size(memory):
    record(memory, 2)
    b = get_stable_batch(memory, 2, 1)
    b.predictions = [[0.0, 0.0]] * 2
    memory.update(b)

