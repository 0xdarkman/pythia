import pytest

from pythia.core.agents.analytical_agent import AnalyticalAgent
from pythia.tests.crypto_doubles import PairEntryStub


def make_agent(dist, diff_thresh, diff_window, targets=None):
    if targets is None:
        targets = ["ETH"]
    return AnalyticalAgent(dist, diff_thresh, diff_window, targets)


@pytest.fixture
def agent():
    return make_agent("0.4", "0.001", 2)


def exchange(pair, rate, fee):
    return {pair: PairEntryStub(pair, rate, 0.1, 0.9, 0.01, fee)}


def exchanges(*rates):
    rates_dict = dict()
    for pair, rate, fee in rates:
        r = exchange(pair, rate, fee)
        rates_dict.update(r)

    return rates_dict


def state(coin, exchange_info):
    return coin, exchange_info


def test_start_no_action(agent):
    assert agent.start(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None


def test_no_records(agent):
    assert agent.step(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None


def test_zero_rates(agent):
    assert agent.step(state("BTC", exchange("BTC_ETH", "0", "0.001"))) is None
    assert agent.step(state("BTC", exchange("BTC_ETH", "0", "0.001"))) is None
    assert agent.step(state("BTC", exchange("BTC_ETH", "0", "0.001"))) is None


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


def test_records_include_starting_state(agent):
    assert agent.start(state("BTC", exchange("BTC_ETH", "10", "0.001"))) is None
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
    assert agent.step(state("ETH", exchanges(("BTC_ETH", "10", "0.0"), ("ETH_BTC", "0.099", "0.0")))) == "BTC"


def test_differential_is_normalized():
    agent = make_agent("0.5", "0.1", 2)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    assert agent.step(state("BTC", exchanges(("BTC_ETH", "0.063", "0.001"), ("ETH_BTC", "15", "0.001")))) == "ETH"


def test_differential_is_not_enough():
    agent = make_agent("0.5", "0.1", 2)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    assert agent.step(state("BTC", exchange("BTC_ETH", "0.0631", "0.001"))) is None


def test_differential_window_falling_steadily():
    agent = make_agent("0.5", "0.1", 3)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.063", "0.001")))
    assert agent.step(state("BTC", exchanges(("BTC_ETH", "0.0567", "0.001"), ("ETH_BTC", "15", "0.001")))) == "ETH"


def test_differential_window_not_falling_enough():
    agent = make_agent("0.5", "0.1", 3)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.063", "0.001")))
    assert agent.step(state("BTC", exchange("BTC_ETH", "0.0568", "0.001"))) is None


def test_differential_window_rising_in_between():
    agent = make_agent("0.5", "0.1", 3)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.071", "0.001")))
    assert agent.step(state("BTC", exchange("BTC_ETH", "0.0567", "0.001"))) is None


def test_differential_window_falling_too_far_in_between():
    agent = make_agent("0.5", "0.1", 3)
    agent.step(state("BTC", exchange("BTC_ETH", "0.01", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.07", "0.001")))
    agent.step(state("BTC", exchange("BTC_ETH", "0.056", "0.001")))
    assert agent.step(state("BTC", exchange("BTC_ETH", "0.0561", "0.001"))) is None


def test_is_monitoring_multiple_targets():
    agent = make_agent("0.5", "0.1", 2, ["ETH", "SALT"])
    agent.step(state("BTC", exchanges(("BTC_ETH", "9", "0.001"), ("BTC_SALT", "0.01", "0.001"))))
    agent.step(state("BTC", exchanges(("BTC_ETH", "8", "0.001"), ("BTC_SALT", "0.07", "0.001"))))
    assert agent.step(state("BTC", exchanges(("BTC_ETH", "8", "0.001"), ("BTC_SALT", "0.063", "0.001"),
                                             ("SALT_ETH", "120", "0.001"), ("SALT_BTC", "15", "0.001")))) == "SALT"


def test_multiple_targets_back_and_forth():
    agent = make_agent("0.4", "0.001", 2, ["ETH", "SALT"])
    agent.step(state("BTC", exchanges(("BTC_ETH", "12", "0.0"), ("BTC_SALT", "0.1", "0.0"))))
    agent.step(state("BTC", exchanges(("BTC_ETH", "11", "0.0"), ("BTC_SALT", "0.15", "0.0"))))
    assert agent.step(state("BTC", exchanges(("BTC_ETH", "10", "0.0"), ("BTC_SALT", "0.141", "0.0"),
                                             ("SALT_ETH", "70", "0.0"), ("SALT_BTC", "7", "0.0")))) == "SALT"
    agent.step(state("SALT", exchanges(("SALT_ETH", "60", "0.001"), ("SALT_BTC", "10.5", "0.0"))))
    assert agent.step(state("SALT", exchanges(("SALT_ETH", "50", "0.0"), ("SALT_BTC", "9.9", "0.0"),
                                              ("BTC_ETH", "5", "0.0"), ("BTC_SALT", "0.1", "0.0")))) == "BTC"


def test_multiple_targets_switch_between_targets():
    agent = make_agent("0.4", "0.001", 2, ["ETH", "SALT"])
    agent.step(state("BTC", exchanges(("BTC_ETH", "12", "0.001"), ("BTC_SALT", "10", "0.001"))))
    agent.step(state("BTC", exchanges(("BTC_ETH", "11", "0.001"), ("BTC_SALT", "15", "0.001"))))
    assert agent.step(state("BTC", exchanges(("BTC_ETH", "10", "0.001"), ("BTC_SALT", "14.001", "0.001"),
                                             ("SALT_ETH", "0.7", "0.001"), ("SALT_BTC", "0.07", "0.001")))) == "SALT"
    agent.step(state("SALT", exchanges(("SALT_ETH", "1.05", "0.001"), ("SALT_BTC", "0.105", "0.001"))))
    assert agent.step(state("SALT", exchanges(("SALT_ETH", "0.99", "0.001"), ("SALT_BTC", "0.106", "0.001"),
                                              ("ETH_SALT", "1", "0.001"), ("ETH_BTC", "0.106", "0.001")))) == "ETH"


def test_finish_does_nothing(agent):
    agent.finish(10)
