from pythia.core.reinforcement.q_neuronal import QNeuronal


class MockAnn:
    def __init__(self, expected_input=None, predicts=None, expected_inputs=None, expected_signals=None):
        self.expected_input = expected_input
        self.predicts = predicts
        self.expected_inputs = expected_inputs
        self.expected_signals = expected_signals
        self.train_was_called = False

    def predict(self, ann_input):
        assert ann_input == self.expected_input
        return self.predicts

    def train(self, input_batch, signal_batch):
        assert list(input_batch) == self.expected_inputs
        assert list(signal_batch) == self.expected_signals
        self.train_was_called = True


def make_q(ann, memory_size=None):
    return QNeuronal(ann, memory_size)


def test_prediction():
    ann = MockAnn(expected_input=[0.5, 0.2, 0.1, 1], predicts=0.3)
    q = make_q(ann)
    assert q[[0.5, 0.2, 0.1], 1] == 0.3


def test_learning():
    ann = MockAnn(expected_inputs=[[0.5, 0.2, 0.1, 1]], expected_signals=[[-0.9]])
    q = make_q(ann)
    q.learn([0.5, 0.2, 0.1], 1, -0.9)
    assert ann.train_was_called


def test_memory_not_filled():
    ann = MockAnn()
    q = make_q(ann, memory_size=2)
    q.learn([0.5, 0.2, 0.1], 1, -0.9)
    assert not ann.train_was_called


def test_train_on_memory_batch():
    ann = MockAnn(expected_inputs=[[0.5, 0.2, 1], [0.1, 0.9, 0]], expected_signals=[[-0.3], [1.7]])
    q = make_q(ann, memory_size=2)
    q.learn([0.5, 0.2], 1, -0.3)
    q.learn([0.1, 0.9], 0, 1.7)
    assert ann.train_was_called
