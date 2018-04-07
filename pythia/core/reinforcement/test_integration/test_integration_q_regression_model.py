import pytest
from pytest import approx

from pythia.core.reinforcement.q_model import InvalidDataSize
from pythia.core.reinforcement.q_regression_model import QRegressionModel

import tensorflow as tf


@pytest.fixture(scope="session", autouse=True)
def config_tensorflow():
    original_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(3)
    tf.set_random_seed(42)
    yield
    tf.logging.set_verbosity(original_v)


@pytest.fixture(autouse=True)
def set_session():
    tf.reset_default_graph()
    with tf.Session():
        yield


@pytest.fixture
def two_in_model():
    return make_model(2)


@pytest.fixture
def deep_model():
    return make_model(1, [1])


def make_model(input_size, hidden=None, lr=0.01):
    if hidden is None:
        hidden = []
    return QRegressionModel(input_size, hidden, lr=lr, seed=7)


def test_predicting_with_wrong_state_size(two_in_model):
    with pytest.raises(InvalidDataSize) as e_info:
        two_in_model.predict([[1, 2, 3]])
    assert str(e_info.value) == "Input size required is 2, but received state has length 3"


def test_training_with_wrong_state_size(two_in_model):
    with pytest.raises(InvalidDataSize) as e_info:
        two_in_model.train([[1, 2, 3]], [[1]])
    assert str(e_info.value) == "Input size required is 2, but received state has length 3"


def test_wrong_action_size(two_in_model):
    with pytest.raises(InvalidDataSize) as e_info:
        two_in_model.train([[1, 2]], [[1, 2]])
    assert str(e_info.value) == "Output size of the model is 1, the training model_data has length 2"


def test_states_and_targets_not_matching(two_in_model):
    with pytest.raises(InvalidDataSize) as e_info:
        two_in_model.train([[1, 2]], [[1], [2]])
    assert str(e_info.value) == "The number of states (1) differs from the numbers of targets (2)"


def test_initialization():
    assert make_model(1).predict([[1]])[0][0] == approx(-0.055630207)


def test_train_gravitates_towards_signal():
    model = make_model(1)
    model.train([[1]], [[1]])
    assert model.predict([[1]])[0][0] == approx(-0.013404997)


def test_learning_rate():
    model = make_model(1, lr=1)
    model.train([[1]], [[1]])
    assert model.predict([[1]])[0][0] == approx(4.1668906)


def test_training_deep_model():
    model = make_model(2, [10, 2], lr=0.1)
    for _ in range(0, 500):
        model.train([[0, 0], [0, 1]] * 30, [[3.0], [-1.5]] * 30)

    assert model.predict([[0, 0]])[0][0] == approx(3.0, rel=1e-2)
    assert model.predict([[0, 1]])[0][0] == approx(-1.5, rel=1e-2)
