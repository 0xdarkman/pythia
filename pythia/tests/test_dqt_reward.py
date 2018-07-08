import pytest

from pythia.core.agents.dqt_agent import DQTRewardCalc
from pythia.tests.dqt_doubles import make_raw_state, exchange


@pytest.mark.parametrize("n,state,reward", [
    (1, make_raw_state([{"SYMA_SYMB": exchange(close=100), "SYMB_SYMA": exchange(close=0.01)},
                        {"SYMA_SYMB": exchange(close=110), "SYMB_SYMA": exchange(close=0.011)}]), [0.9, 1, 1.1]),
    (1, make_raw_state([{"SYMA_SYMB": exchange(close=100), "SYMB_SYMA": exchange(close=0.01)},
                        {"SYMA_SYMB": exchange(close=90), "SYMB_SYMA": exchange(close=0.009)}]), [1.1, 1, 0.9]),
    (2, make_raw_state([{"SYMA_SYMB": exchange(close=110), "SYMB_SYMA": exchange(close=0.011)},
                        {"SYMA_SYMB": exchange(close=100), "SYMB_SYMA": exchange(close=0.01)},
                        {"SYMA_SYMB": exchange(close=90), "SYMB_SYMA": exchange(close=0.009)}]),
     [1.1 * 100 / 110, 100 / 110, 0.9 * 100 / 110]),
])
def test_reward_calculation(n, state, reward):
    r = DQTRewardCalc(n, "SYMA_SYMB")
    assert r(state) == pytest.approx(reward)
