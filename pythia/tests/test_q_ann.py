import unittest

import numpy as np

from pythia.reinforcement.q_ann import QAnn
from pythia.reinforcement.q_function import InvalidAction


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


class QAnnTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelFake()

    def test_action_out_of_action_space(self):
        qann = self.make_qann()
        with self.assertRaises(InvalidAction) as cm:
            unused = qann[[0, 0], 5]
        self.assertEqual("The action 5 is not part of action space [1, 2]", cm.exception.args[0])

    def make_qann(self, memory_size=1):
        return QAnn(self.model, [1, 2], memory_size)

    def test_returns_prediction_from_model(self):
        self.model.prediction = 3
        qann = self.make_qann()
        self.assertEqual(3, qann[[1, 4], 2])
        self.assertTrue(np.array_equal([[1, 4, 2]], self.model.received_state))

    def test_learn_invalid_action(self):
        qann = self.make_qann()
        with self.assertRaises(InvalidAction) as cm:
            qann.learn([0, 0], 0, 0)
        self.assertEqual("The action 0 is not part of action space [1, 2]", cm.exception.args[0])

    def test_no_training_when_memory_is_not_full(self):
        qann = self.make_qann(2)
        qann.learn([0, 0], 1, 0)
        self.assertIsNone(self.model.received_feature_batch)
        self.assertIsNone(self.model.received_target_batch)

    def test_training_batch(self):
        qann = self.make_qann(2)
        qann.learn([0, 0], 1, 0)
        qann.learn([0, 1], 2, 3)
        self.assertTrue(np.array_equal([[0, 0, 1], [0, 1, 2]], self.model.received_feature_batch),
                        self.model.received_feature_batch)
        self.assertTrue(np.array_equal([[0], [3]], self.model.received_target_batch), self.model.received_target_batch)

    def test_memory_is_first_in_last_out(self):
        qann = self.make_qann(2)
        qann.learn([0, 0], 1, 0)
        qann.learn([0, 1], 2, 3)
        qann.learn([1, 0], 1, -1)
        self.assertTrue(np.array_equal([[0, 1, 2], [1, 0, 1]], self.model.received_feature_batch),
                        self.model.received_feature_batch)
        self.assertTrue(np.array_equal([[3], [-1]], self.model.received_target_batch), self.model.received_target_batch)


if __name__ == '__main__':
    unittest.main()
