import os
import shutil

import pytest

from pythia.core.agents.fpm_memory import FPMMemory
from pythia.tests.fpm_doubles import Prices
from pythia.tests.fpm_memory_test_functions import assert_states, assert_batch, record, batch, get_stable_batch


class FPMMemorySUT(FPMMemory):
    def record(self, prices, portfolio):
        super().record(prices.to_array(), portfolio)


class ConfigBuilder:
    def __init__(self):
        self.cfg = {"training": {"window": 2, "size": 10, "beta": 0.1}, "trading": {"coins": ["SYM1"]}}

    def size(self, value):
        self.cfg["training"]["size"] = value
        return self

    def num_symbols(self, value):
        self.cfg["trading"]["coins"] = ["SYM{}".format(i) for i in range(1, value + 1)]
        return self

    def window(self, value):
        self.cfg["training"]["window"] = value
        return self


@pytest.fixture
def save_file():
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data/test_save_memory/point.npz")
    yield file
    d = os.path.dirname(file)
    if os.path.exists(d):
        shutil.rmtree(d)


@pytest.fixture
def default():
    return ConfigBuilder()


def make_memory(cfg=None):
    return FPMMemorySUT(ConfigBuilder().cfg if cfg is None else cfg)


def test_fpm_memory_restores_latest_record(save_file):
    m_saved = make_memory()
    m_saved.record(Prices({"SYM1": {"high": 4.0, "low": 1.0, "close": 1.5}}), [1.0, 0.0])
    m_saved.record(Prices({"SYM1": {"high": 2.5, "low": 1.2, "close": 2.0}}), [1.0, 0.0])
    m_saved.record(Prices({"SYM1": {"high": 2.0, "low": 0.5, "close": 1.0}}), [0.0, 1.0])
    m_saved.save(save_file)
    assert_states([[[2.0, 1.0]], [[2.5, 2.0]], [[1.2, 0.5]]], [1.0], *m_saved.get_latest())

    m_restored = make_memory()
    m_restored.restore(save_file)
    assert_states([[[2.0, 1.0]], [[2.5, 2.0]], [[1.2, 0.5]]], [1.0], *m_restored.get_latest())


def test_fpm_memory_restores_with_correctly_when_exceeding_capacity(save_file, default):
    cfg = default.window(1).size(2).cfg

    m_saved = make_memory(cfg)
    record(m_saved, 3)
    m_saved.save(save_file)
    assert_batch(batch(2.0, 3.0), get_stable_batch(m_saved, 1, 11))

    m_restored = make_memory(cfg)
    m_restored.restore(save_file)
    assert_batch(batch(2.0, 3.0), get_stable_batch(m_restored, 1, 11))


@pytest.mark.parametrize("saved_cfg, restored_cfg", [
    (ConfigBuilder().size(2).cfg, ConfigBuilder().size(1).cfg),
    (ConfigBuilder().num_symbols(1).cfg, ConfigBuilder().num_symbols(2).cfg),
    (ConfigBuilder().window(2).cfg, ConfigBuilder().window(3).cfg)
])
def test_fpm_memory_raises_error_when_restored_memory_does_not_fit_configuration(save_file, saved_cfg, restored_cfg):
    m_saved = make_memory(saved_cfg)
    m_saved.save(save_file)
    m_restored = make_memory(restored_cfg)
    with pytest.raises(FPMMemory.RestorationError):
        m_restored.restore(save_file)
