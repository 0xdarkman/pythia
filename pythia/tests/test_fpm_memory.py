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
            t.append([v['high'], v['low'], v['close']])
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

    @property
    def memory(self):
        if self._memory is None:
            self._memory = FPMMemory(self.window, 1000, self.beta)
        return self._memory


@pytest.fixture
def memory():
    return MemoryTestBuilder()


def environment_input(identifier):
    w0 = 1.0 / identifier
    return Prices({"SYM": {"high": identifier, "low": 1.0, "close": 1.0}}), [w0, 1 - w0]


def state(identifier):
    w0 = 1.0 / identifier
    return [[[identifier]], [[1.0]], [[1.0]]], [w0, 1 - w0]


def identify_state(state):
    prices, _ = state
    return int(prices[0][0][0])


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


def test_nothing_recorded_latest_is_none(memory):
    pr, po = memory.get_latest()
    assert pr is None and po is None


@pytest.mark.parametrize("price, portfolio, price_tensor", [
    (Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 2.0}}), [1.0, 0.0], [[[2.0]], [[0.5]], [[1.0]]]),
    (Prices({"SYM2": {"high": 2.0, "low": 0.1, "close": 0.5}}), [0.5, 0.5], [[[4.0]], [[0.2]], [[1.0]]]),
])
def test_enough_records_latest_returns_last_price_tensor_and_portfolio(memory, price, portfolio, price_tensor):
    memory.record(price, portfolio)
    assert_states(price_tensor, portfolio, *memory.get_latest())


def test_multiple_assets(memory):
    p = Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 2.0}, "SYM2": {"high": 2.0, "low": 0.1, "close": 0.5}})
    memory.record(p, [1.0, 0.0, 0.0])
    assert_states([[[2.0], [4.0]], [[0.5], [0.2]], [[1.0], [1.0]]], [1.0, 0.0, 0.0], *memory.get_latest())


def test_raise_data_mismatch_error_when_amount_of_symbols_does_not_fit_portfolio_vector(memory):
    p = Prices({"SYM1": {"high": 2.0, "low": 1.0, "close": 1.5}, "SYM2": {"high": 1.0, "low": 0.1, "close": 0.8}})
    with pytest.raises(FPMMemory.DataMismatchError):
        memory.record(p, [1.0, 0.0])


def test_latest_returns_none_if_there_are_not_enough_records_to_fill_price_window(memory):
    memory.window = 2
    memory.record(Prices({"SYM1": {"high": 2.0, "low": 1.0, "close": 1.5}}), [1.0, 0.0])
    assert_states(None, None, *memory.get_latest())


def test_price_vector_contains_price_history_quotient_of_most_recent_price_in_window(memory):
    memory.window = 2
    memory.record(Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 1.5}}), [1.0, 0.0])
    memory.record(Prices({"SYM1": {"high": 2.5, "low": 1.2, "close": 3.0}}), [1.0, 0.0])
    assert_states([[[4.0 / 3.0, 2.5 / 3.0]], [[1.0 / 3.0, 1.2 / 3.0]], [[1.5 / 3.0, 3.0 / 3.0]]], [1.0, 0.0],
                  *memory.get_latest())


def test_window_with_multiple_assets(memory):
    memory.window = 2
    p1 = Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 1.0}, "SYM2": {"high": 2.0, "low": 0.1, "close": 1.0}})
    p2 = Prices({"SYM1": {"high": 2.0, "low": 0.5, "close": 2.0}, "SYM2": {"high": 3.0, "low": 0.2, "close": 0.5}})
    memory.record(p1, [1.0, 0.0, 0.0])
    memory.record(p2, [1.0, 0.0, 0.0])
    assert_states([[[2.0, 1.0], [4.0, 6.0]], [[0.5, 0.25], [0.2, 0.4]], [[0.5, 1.0], [2.0, 1.0]]], [1.0, 0.0, 0.0],
                  *memory.get_latest())


def test_drop_price_information_after_window(memory):
    memory.window = 2
    memory.record(Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 1.5}}), [1.0, 0.0])
    memory.record(Prices({"SYM1": {"high": 2.5, "low": 1.2, "close": 2.0}}), [1.0, 0.0])
    memory.record(Prices({"SYM1": {"high": 2.0, "low": 0.5, "close": 1.0}}), [1.0, 0.0])
    assert_states([[[2.5, 2.0]], [[1.2, 0.5]], [[2.0, 1.0]]], [1.0, 0.0], *memory.get_latest())


def test_return_empty_batch_when_nothing_is_recorded(memory):
    assert (memory.get_random_batch(1) == np.array([])).all()


def test_return_trivial_random_batch(memory):
    memory.record(*environment_input(3.0))
    assert_states(*state(3.0), *memory.get_random_batch(1)[0])


def test_select_randomly_from_history(memory):
    np.random.seed(11)
    memory.record(*environment_input(1.0))
    memory.record(*environment_input(2.0))
    memory.record(*environment_input(3.0))
    assert_states(*state(2.0), *memory.get_random_batch(1)[0])


def test_random_batches_preserve_time_series(memory):
    np.random.seed(7)
    memory.record(*environment_input(1.0))
    memory.record(*environment_input(2.0))
    memory.record(*environment_input(3.0))
    batch = memory.get_random_batch(2)
    assert_states(*state(2.0), *batch[0])
    assert_states(*state(3.0), *batch[1])


def test_not_enough_data_to_fill_batch_size_truncates_the_batch(memory):
    np.random.seed(7)
    memory.record(*environment_input(1.0))
    memory.record(*environment_input(2.0))
    batch = memory.get_random_batch(2)
    assert len(batch) == 2
    assert_states(*state(1.0), *batch[0])
    assert_states(*state(2.0), *batch[1])


def test_batch_selection_follows_a_geometrically_decaying_distribution(memory):
    np.random.seed(7)
    memory.beta = 0.5
    records = 5
    record(memory, 5)

    distribution = [0] * (records - 1)
    n = 1000
    for _ in range(0, n):
        distribution[identify_state(memory.get_random_batch(2)[0]) - 1] += 1

    distribution[:] = [p / n for p in distribution]
    assert pytest.approx([0.125, 0.125, 0.25, 0.5], 0.1) == distribution


def test_portfolio_weights_of_a_batch_can_be_updated(memory):
    seed = int(time.time())  # should be stable in a random environment so select a random seed
    record(memory, 100)

    batch = get_stable_batch(memory, 3, seed)
    batch.weights = [[0.0, 0.0]] * 3
    memory.update(batch)

    batch = get_stable_batch(memory, 3, seed)
    assert_weights([[0.0, 0.0]] * 3, batch.weights)
