import numpy as np
import pytest

from pythia.reinforcement.q_model import InvalidDataSize
from pythia.reinforcement.q_regression_model import QRegressionModel


def make_model(input_size, hidden=None):
    if hidden is None:
        hidden = [1]
    return QRegressionModel(input_size, hidden)


@pytest.fixture
def two_in_model():
    return make_model(2)


@pytest.fixture
def deep_model():
    return make_model(1, [1])


def test_predicting_with_wrong_state_size(two_in_model):
    with pytest.raises(InvalidDataSize) as e_info:
        two_in_model.predict(np.array([[1, 2, 3]]))
    assert str(e_info.value) == "Input size required is 2, but received state has length 3"


def test_training_with_wrong_state_size(two_in_model):
    with pytest.raises(InvalidDataSize) as e_info:
        two_in_model.train(np.array([[1, 2, 3]]), np.array([[1]]))
    assert str(e_info.value) == "Input size required is 2, but received state has length 3"


def test_wrong_action_size(two_in_model):
    with pytest.raises(InvalidDataSize) as e_info:
        two_in_model.train(np.array([[1, 2]]), np.array([[1, 2]]))
    assert str(e_info.value) == "Output size of the model is 1, the training model_data has length 2"


def test_states_and_targets_not_matching(two_in_model):
    with pytest.raises(InvalidDataSize) as e_info:
        two_in_model.train(np.array([[1, 2]]), np.array([[1], [2]]))
    assert str(e_info.value) == "The number of states (1) differs from the numbers of targets (2)"


def test_initialization(deep_model):
    assert deep_model.predict(np.array([[1]]))[0] is not None


def test_train(deep_model):
    deep_model.train(np.array([[1]]), np.array([[1]]))
    assert deep_model.predict(np.array([[1]]))[0] != 0
