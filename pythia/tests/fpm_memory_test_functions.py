import numpy as np

from pythia.core.agents.fpm_memory import FPMMemory
from pythia.tests.fpm_doubles import Prices


def assert_states(expected_prices, expected_portfolio, actual_prices, actual_portfolio):
    assert (np.array(expected_prices) == actual_prices).all()
    assert (np.array(expected_portfolio) == actual_portfolio).all()


def assert_batch(expected, actual):
    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        ep, ew, ef = e
        ap, aw, af = a
        assert ep.shape == ap.shape
        assert ew.shape == aw.shape
        assert (ep == ap).all()
        assert (ew == aw).all()
        assert (ef == af).all()


def environment_input(identifier):
    w0 = 1.0 / identifier
    return Prices({"SYM": {"high": 1.0, "low": 1.0, "close": identifier}}), [w0, 1 - w0]


def state(identifier):
    w0 = 1.0 / identifier
    return [[[1.0]], [[w0]], [[w0]]], [1 - w0]


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
