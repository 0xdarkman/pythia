import unittest

import numpy as np

from pythia.reinforcement.q_model import InvalidDataSize
from pythia.reinforcement.q_regression_model import QRegressionModel


class QRegressionModelTests(unittest.TestCase):
    def test_wrong_state_size(self):
        model = self.make_model(2)
        with self.assertRaises(InvalidDataSize) as cm1:
            model.predict(np.array([[1, 2, 3]]))
        self.assertEqual("Input size required is 2, but received state has length 3", cm1.exception.args[0])
        with self.assertRaises(InvalidDataSize) as cm2:
            model.train(np.array([[1, 2, 3]]), np.array([[1]]))
        self.assertEqual("Input size required is 2, but received state has length 3", cm2.exception.args[0])

    @staticmethod
    def make_model(input_size, hidden=None):
        if hidden is None:
            hidden = [1]
        return QRegressionModel(input_size, hidden)

    def test_wrong_action_size(self):
        model = self.make_model(2)
        with self.assertRaises(InvalidDataSize) as cm:
            model.train(np.array([[1, 2]]), np.array([[1, 2]]))
        self.assertEqual("Output size of the model is 1, the training model_data has length 2", cm.exception.args[0])

    def test_states_and_targets_not_matching(self):
        model = self.make_model(2)
        with self.assertRaises(InvalidDataSize) as cm:
            model.train(np.array([[1, 2]]), np.array([[1], [2]]))
        self.assertEqual("The number of states (1) differs from the numbers of targets (2)", cm.exception.args[0])

    def test_initialization(self):
        model = self.make_model(1, [1])
        self.assertIsNotNone(model.predict(np.array([[1]]))[0])

    def test_train(self):
        model = self.make_model(1, [1])
        model.train(np.array([[1]]), np.array([[1]]))
        self.assertNotAlmostEqual(0, model.predict(np.array([[1]]))[0])


if __name__ == '__main__':
    unittest.main()
