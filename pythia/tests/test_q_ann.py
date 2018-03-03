import numpy as np
import pytest

from pythia.core.reinforcement.q_ann import QAnn
from pythia.core.reinforcement.q_function import InvalidAction


class ModelFake(object):
    def __init__(self):
        self.received_state = None
        self.received_feature_batch = None
        self.received_target_batch = None
        self.prediction = 0

    def predict(self, state):
        self.received_state = state
        return [self.prediction]

    def train(self, features, targets):
        self.received_feature_batch = features
        self.received_target_batch = targets


@pytest.fixture()
def qann():
    """"Creates a default QAnn object with a fake model, actions 1 and 2 and one memory cell"""
    return QAnn(ModelFake(), [1, 2], 1)


@pytest.fixture()
def qann_with_memory():
    """Creates a QAnn object with a fake model, actions 1 and 2 and two memory cells"""
    return QAnn(ModelFake(), [1, 2], 2)


def test_action_out_of_action_space(qann):
    with pytest.raises(InvalidAction) as e_info:
        unused = qann[[0, 0], 5]
    assert e_info.value.args[0] == "The action 5 is not part of action space [1, 2]"


def test_returns_prediction_from_model(qann):
    qann.model.prediction = 3
    assert qann[[1, 4], 2] == 3


def test_hands_state_and_action_to_model(qann):
    unused = qann[[1, 4], 2]
    assert np.array_equal([[1, 4, 2]], qann.model.received_state)


def test_learn_invalid_action(qann):
    with pytest.raises(InvalidAction) as e_info:
        qann.learn([0, 0], 0, 0)
    assert e_info.value.args[0] == "The action 0 is not part of action space [1, 2]"


def test_no_training_when_memory_is_not_full(qann_with_memory):
    qann_with_memory.learn([0, 0], 1, 0)
    assert qann_with_memory.model.received_feature_batch is None
    assert qann_with_memory.model.received_target_batch is None


def test_training_batch(qann_with_memory):
    qann_with_memory.learn([0, 0], 1, 0)
    qann_with_memory.learn([0, 1], 2, 3)
    assert np.array_equal(qann_with_memory.model.received_feature_batch, [[0, 0, 1], [0, 1, 2]])
    assert np.array_equal(qann_with_memory.model.received_target_batch, [[0], [3]])


def test_memory_is_first_in_last_out(qann_with_memory):
    qann_with_memory.learn([0, 0], 1, 0)
    qann_with_memory.learn([0, 1], 2, 3)
    qann_with_memory.learn([1, 0], 1, -1)
    assert np.array_equal(qann_with_memory.model.received_feature_batch, [[0, 1, 2], [1, 0, 1]])
    assert np.array_equal(qann_with_memory.model.received_target_batch, [[3], [-1]])
