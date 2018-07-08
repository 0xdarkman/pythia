import pytest

from pythia.core.agents.dqt_agent import DQTDiffStateTransformer, DQTRatioStateTransformer
from pythia.tests.dqt_doubles import exchange, make_raw_state


def transform_diff(raw):
    t = DQTDiffStateTransformer("SYMA_SYMB")
    return t(raw)


def transform_ratio(raw):
    t = DQTRatioStateTransformer("SYMA_SYMB")
    return t(raw)


@pytest.mark.parametrize("raw, array", [
    (make_raw_state([{"SYMA_SYMB": exchange(close=100), "SYMB_SYMA": exchange(close=0.01)},
                     {"SYMA_SYMB": exchange(close=110), "SYMB_SYMA": exchange(close=0.011)}]), [10]),
    (make_raw_state([{"SYMA_SYMB": exchange(close=100), "SYMB_SYMA": exchange(close=0.01)},
                     {"SYMA_SYMB": exchange(close=110), "SYMB_SYMA": exchange(close=0.011)},
                     {"SYMA_SYMB": exchange(close=90), "SYMB_SYMA": exchange(close=1.0 / 90.0)}]), [10, -20])
])
def test_transforms_raw_input_to_diff_state_array(raw, array):
    assert (transform_diff(raw) == array).all()

@pytest.mark.parametrize("raw, array", [
    (make_raw_state([{"SYMA_SYMB": exchange(close=100), "SYMB_SYMA": exchange(close=0.01)},
                     {"SYMA_SYMB": exchange(close=110), "SYMB_SYMA": exchange(close=0.011)}]), [1.1]),
    (make_raw_state([{"SYMA_SYMB": exchange(close=100), "SYMB_SYMA": exchange(close=0.01)},
                     {"SYMA_SYMB": exchange(close=110), "SYMB_SYMA": exchange(close=0.011)},
                     {"SYMA_SYMB": exchange(close=90), "SYMB_SYMA": exchange(close=1.0 / 90.0)}]), [1.1, 0.9])
])
def test_transform_raw_input_to_ratio_state_array(raw, array):
    assert (transform_ratio(raw) == array).all()
