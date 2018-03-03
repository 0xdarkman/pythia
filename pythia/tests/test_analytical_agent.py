import pytest

from pythia.core.agents.analytical_agent import AnalyticalAgent
from pythia.tests.doubles import PairEntryStub


@pytest.fixture
def agent():
    return AnalyticalAgent("0.4", "0.001", 2)


def exchange(pair, rate, minerFee):
    return {pair: PairEntryStub(pair, rate, 0.1, 0.9, 0.01, minerFee)}


def exchanges(*rates):
    rates_dict = dict()
    for pair, rate, minerFee in rates:
        r = exchange(pair, rate, minerFee)
        rates_dict.update(r)

    return rates_dict


def state(coin, exchange_info):
    return coin, exchange_info


def test_no_records(agent):
    assert agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None


def test_records_but_no_change(agent):
    assert agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None
    assert agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None
    assert agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None


def test_positive_rate_differential_but_distance_too_low(agent):
    assert agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None
    assert agent.step(state("BTC", exchange("BTC_ETH", "15", "0.001"))) is None
    assert agent.step(state("BTC", exchange("BTC_ETH", "14", "0.001"))) is None


def test_distance_reached_but_no_positive_differential(agent):
    assert agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None
    assert agent.step(state("BTC", exchange("BTC_ETH", "14", "0.001"))) is None
    assert agent.step(state("BTC", exchange("BTC_ETH", "15", "0.001"))) is None


def test_records_indicate_exchange_opportunity(agent):
    assert agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None
    assert agent.step(state("BTC", exchange("BTC_ETH", "15", "0.001"))) is None
    assert agent.step(state("BTC", exchanges(("BTC_ETH", "14.001", "0.001"), ("ETH_BTC", "0.07", "0.001")))) == "ETH"


def test_switch_rates_after_exchange(agent):
    agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "15", "0.001")))
    agent.step(state("BTC", exchanges(("BTC_ETH", "14.001", "0.001"), ("ETH_BTC", "0.07", "0.001"))))
    assert agent.step(state("ETH", exchange("ETH_BTC", "0.105", "0.001"))) is None


def test_switch_back_and_forth(agent):
    agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "15", "0.001")))
    assert agent.step(state("BTC", exchanges(("BTC_ETH", "14.001", "0.001"), ("ETH_BTC", "0.07", "0.001")))) == "ETH"
    agent.step(state("ETH", exchange("ETH_BTC", "0.105", "0.001")))
    assert agent.step(state("ETH", exchanges(("BTC_ETH", "10", "0.001"), ("ETH_BTC", "0.099", "0.001")))) == "BTC"


def test_differential_is_normalized():
    agent = AnalyticalAgent("0.5", "0.1", 2)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    assert agent.step(state("BTC", exchanges(("BTC_ETH", "0.063", "0.001"), ("ETH_BTC", "15", "0.001")))) == "ETH"


def test_differential_is_not_enough():
    agent = AnalyticalAgent("0.5", "0.1", 2)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    assert agent.step(state("BTC", exchange("BTC_ETH", "0.0631", "0.001"))) is None


def test_differential_window_falling_steadily():
    agent = AnalyticalAgent("0.5", "0.1", 3)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.063", "0.001")))
    assert agent.step(state("BTC",exchanges(("BTC_ETH", "0.0567", "0.001"), ("ETH_BTC", "15", "0.001")))) == "ETH"


def test_differential_window_not_falling_enough():
    agent = AnalyticalAgent("0.5", "0.1", 3)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.063", "0.001")))
    assert agent.step(state("BTC", exchange("BTC_ETH", "0.0568", "0.001"))) is None


def test_differential_window_rising_in_between():
    agent = AnalyticalAgent("0.5", "0.1", 3)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.071", "0.001")))
    assert agent.step(state("BTC", exchange("BTC_ETH", "0.0567", "0.001"))) is None


def test_differential_window_falling_too_far_in_between():
    agent = AnalyticalAgent("0.5", "0.1", 3)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.056", "0.001")))
    assert agent.step(state("BTC", exchange("BTC_ETH", "0.0561", "0.001"))) is None
