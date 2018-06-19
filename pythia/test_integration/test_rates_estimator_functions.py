import pytest

from tensorflow.python.feature_column.feature_column import numeric_column

from pythia.core.models.rates_estimator_functions import make_feature_columns, process_input_states, \
    process_reward_targets


@pytest.mark.parametrize("input_size,expected", [pytest.param(3, [numeric_column("token"),
                                                                  numeric_column("balance"),
                                                                  numeric_column("price_0")]),
                                                 pytest.param(4, [numeric_column("token"),
                                                                  numeric_column("balance"),
                                                                  numeric_column("price_0"),
                                                                  numeric_column("price_1")])])
def test_generate_feature_columns_from_input_size(input_size, expected):
    assert make_feature_columns(input_size) == expected


@pytest.mark.parametrize("states,expected", [pytest.param([[0, 0.0, 0.1, 0.2],
                                                           [1, 0.5, 0.2, 0.3]], {"token": [0, 1],
                                                                                 "balance": [0.0, 0.5],
                                                                                 "price_0": [0.1, 0.2],
                                                                                 "price_1": [0.2, 0.3]}),
                                             pytest.param([[1, 0.1, 0.9],
                                                           [0, 0.2, 0.8]], {"token": [1, 0],
                                                                            "balance": [0.1, 0.2],
                                                                            "price_0": [0.9, 0.8]})])
def test_process_input_states(states, expected):
    assert process_input_states(states) == expected


@pytest.mark.parametrize("targets,expected", [pytest.param([[10.0], [-3.5]], {"q_value": [10.0, -3.5]}),
                                              pytest.param([[-1.0], [13.5]], {"q_value": [-1.0, 13.5]})])
def test_process_reward_targets(targets, expected):
    assert process_reward_targets(targets) == expected
